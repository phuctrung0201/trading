"""Simple backtester that evaluates a strategy on OHLCV data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class Result:
    """Container for backtest metrics."""

    profit: float
    profit_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    sharpe_ratio: float
    equity_curve: pd.Series
    strategy_returns: pd.Series = None
    print_interval: str | None = None

    def print(self) -> None:
        """Print a formatted summary of the backtest result."""
        if self.print_interval is None:
            self._print_summary("Backtest Result", self.equity_curve, self.strategy_returns)
        else:
            self._print_by_interval()

    def _print_summary(
        self,
        title: str,
        equity: pd.Series,
        returns: pd.Series,
        capital: float | None = None,
    ) -> None:
        """Print metrics for an equity curve segment."""
        base = capital if capital is not None else equity.iloc[0]
        end_equity = equity.iloc[-1]
        profit = end_equity - base
        profit_pct = (profit / base) * 100

        running_max = equity.cummax()
        drawdown = equity - running_max
        drawdown_pct = (drawdown / running_max) * 100
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()

        is_in_dd = drawdown < 0
        streak_groups = (~is_in_dd).cumsum()
        if is_in_dd.any():
            max_dd_dur = int(is_in_dd.groupby(streak_groups).sum().max())
        else:
            max_dd_dur = 0

        mean_ret = returns.mean()
        std_ret = returns.std()
        # Estimate bars per day from the index
        if len(equity) > 1:
            total_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
            bar_seconds = total_seconds / (len(equity) - 1)
            periods_per_day = 86400 / bar_seconds if bar_seconds > 0 else 1
        else:
            periods_per_day = 1
        annualisation = np.sqrt(365 * periods_per_day)
        sharpe = (mean_ret / std_ret) * annualisation if std_ret != 0 else 0.0

        print(
            f"{title}\n"
            f"  Profit:                {profit:,.2f}  ({profit_pct:+.2f}%)\n"
            f"  Max Drawdown:          {max_dd:,.2f}  ({max_dd_pct:+.2f}%)\n"
            f"  Max Drawdown Duration: {max_dd_dur} bars\n"
            f"  Sharpe Ratio:          {sharpe:.4f}"
        )

    def _print_by_interval(self) -> None:
        """Print cumulative metrics at the end of each interval period."""
        equity = self.equity_curve
        returns = self.strategy_returns
        capital = equity.iloc[0]

        periods = equity.resample(self.print_interval).last().dropna().index

        for period in periods:
            # Cumulative slice from the start up to the end of this period
            eq_slice = equity.loc[:period]
            ret_slice = returns.loc[:period]

            if len(eq_slice) < 2:
                continue

            label = str(period.date()) if hasattr(period, "date") else str(period)
            self._print_summary(f"Backtest Result [{label}]", eq_slice, ret_slice, capital)
            print()



def evaluate(
    ohlc: pd.DataFrame,
    strategies,
    capital: float = 1000.0,
    print_interval: str | None = None,
) -> Result:
    """Run a backtest and return performance metrics.

    The backtest assumes:
    - Positions are taken at the **close** of each bar.
    - Returns are computed bar-to-bar on the close price.
    - Long position (+1) profits when price goes up.
    - Short position (−1) profits when price goes down.

    Strategies are applied as a pipeline in reverse list order: the last
    strategy generates base signals, and earlier strategies act as
    modifiers/overlays (e.g. stop-loss).

    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLCV DataFrame with a ``close`` column.
    strategies : list
        List of strategy instances that implement ``generate_signals(df)``.
        The last entry is the primary signal generator; earlier entries
        are applied as overlays (highest priority first).
    capital : float
        Starting capital.

    Returns
    -------
    Result
        Backtest metrics including profit, max drawdown, Sharpe ratio,
        and the full equity curve.
    """
    # The last strategy is the base signal generator.
    # Preceding strategies are modifiers applied in list order.
    signals = strategies[-1].generate_signals(ohlc)
    for strategy in strategies[:-1]:
        signals = strategy.generate_signals(signals)

    # Bar-to-bar percentage returns on the close price
    close_returns = signals["close"].pct_change().fillna(0)

    # Strategy returns: position from the *previous* bar × this bar's return
    strategy_returns = signals["position"].shift(1).fillna(0) * close_returns

    # Build equity curve
    equity_curve = capital * (1 + strategy_returns).cumprod()

    # ---- Profit ----
    final_equity = equity_curve.iloc[-1]
    profit = final_equity - capital
    profit_pct = (profit / capital) * 100

    # ---- Max Drawdown ----
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max  # always <= 0
    drawdown_pct = (drawdown / running_max) * 100  # as percentage of peak
    max_drawdown = drawdown.min()  # most negative value (absolute)
    max_drawdown_pct = drawdown_pct.min()  # most negative value (percentage)

    # Max drawdown duration: longest streak of consecutive bars below the peak
    is_in_drawdown = drawdown < 0
    # Label each consecutive drawdown streak, then find the longest one
    streak_groups = (~is_in_drawdown).cumsum()
    if is_in_drawdown.any():
        max_drawdown_duration = int(
            is_in_drawdown.groupby(streak_groups).sum().max()
        )
    else:
        max_drawdown_duration = 0

    # ---- Sharpe Ratio ----
    # Annualised assuming 48 half-hour bars per day (30m candles)
    periods_per_day = 48
    annualisation_factor = np.sqrt(365 * periods_per_day)
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe_ratio = (
        (mean_return / std_return) * annualisation_factor
        if std_return != 0
        else 0.0
    )

    return Result(
        profit=profit,
        profit_pct=profit_pct,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration=max_drawdown_duration,
        sharpe_ratio=sharpe_ratio,
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        print_interval=print_interval,
    )
