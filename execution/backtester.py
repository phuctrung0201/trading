"""Simple backtester that evaluates a strategy on OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    prices: pd.Series = None
    strategy_returns: pd.Series = None
    sharpe_window: int | None = None

    def print(self) -> None:
        """Print a formatted summary of the backtest result."""
        self._print_summary("Backtest Result", self.equity_curve, self.strategy_returns)

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

    def plot(self, path: str | None = None) -> None:
        """Plot the backtest equity curve and drawdown.

        Parameters
        ----------
        path : str, optional
            If provided, save the figure to this file path instead of showing it.
        """
        equity = self.equity_curve
        returns = self.strategy_returns if self.strategy_returns is not None else equity.pct_change().fillna(0)
        running_max = equity.cummax()
        drawdown_pct = ((equity - running_max) / running_max) * 100

        # Rolling annualized Sharpe ratio
        if len(equity) > 1:
            total_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
            bar_seconds = total_seconds / (len(equity) - 1)
            periods_per_day = 86400 / bar_seconds if bar_seconds > 0 else 1
        else:
            periods_per_day = 1
        annualisation = np.sqrt(365 * periods_per_day)
        window = self.sharpe_window
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        rolling_std = returns.rolling(window=window, min_periods=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * annualisation
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(
            4, 1, figsize=(14, 11), sharex=True,
            gridspec_kw={"height_ratios": [2, 3, 1, 1]},
        )
        fig.suptitle("Backtest Result", fontsize=14, fontweight="bold")

        # --- Price ---
        if self.prices is not None:
            ax0.plot(self.prices.index, self.prices, color="#9C27B0", linewidth=1.0, label="Close Price")
            ax0.fill_between(self.prices.index, self.prices.min(), self.prices, alpha=0.06, color="#9C27B0")
            ax0.set_ylabel("Price")
            ax0.legend(loc="upper left", fontsize=9)
            ax0.grid(True, alpha=0.3)
        else:
            ax0.set_visible(False)

        # --- Equity curve ---
        ax1.plot(equity.index, equity, color="#2196F3", linewidth=1.2, label="Equity")
        ax1.fill_between(equity.index, equity.iloc[0], equity, alpha=0.08, color="#2196F3")
        ax1.axhline(equity.iloc[0], color="grey", linestyle="--", linewidth=0.8, label="Starting Capital")
        ax1.set_ylabel("Equity")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Annotate max and min equity
        max_equity = equity.max()
        min_equity = equity.min()
        ax1.annotate(
            f"Max Equity: {max_equity:,.2f}",
            xy=(0.99, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold",
            color="#4CAF50",
        )
        ax1.annotate(
            f"Min Equity: {min_equity:,.2f}",
            xy=(0.99, 0.89), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold",
            color="#F44336",
        )
        # --- Rolling Sharpe ratio ---
        ax2.plot(rolling_sharpe.index, rolling_sharpe, color="#FF9800", linewidth=1.0, label=f"Rolling Sharpe ({window}-bar)")
        ax2.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        max_sharpe = rolling_sharpe.dropna().max() if rolling_sharpe.dropna().any() else 0.0
        sharpe_color = "#4CAF50" if max_sharpe >= 0 else "#F44336"
        sharpe_ann_color = "#4CAF50" if self.sharpe_ratio >= 0 else "#F44336"
        ax2.annotate(
            f"Annualized: {self.sharpe_ratio:.4f}",
            xy=(0.99, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold",
            color=sharpe_ann_color,
        )
        ax2.annotate(
            f"Max Rolling: {max_sharpe:.4f}",
            xy=(0.99, 0.85), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold",
            color=sharpe_color,
        )
        ax2.fill_between(
            rolling_sharpe.index, 0, rolling_sharpe,
            where=rolling_sharpe >= 0, alpha=0.15, color="#4CAF50",
        )
        ax2.fill_between(
            rolling_sharpe.index, 0, rolling_sharpe,
            where=rolling_sharpe < 0, alpha=0.15, color="#F44336",
        )
        ax2.set_ylabel("Sharpe Ratio")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # --- Drawdown ---
        ax3.fill_between(drawdown_pct.index, 0, drawdown_pct, color="#F44336", alpha=0.35)
        ax3.plot(drawdown_pct.index, drawdown_pct, color="#F44336", linewidth=0.8)
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Time")
        ax3.grid(True, alpha=0.3)

        # Annotate max drawdown
        ax3.annotate(
            f"Max DD: {self.max_drawdown:,.2f} ({self.max_drawdown_pct:+.2f}%)",
            xy=(0.99, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9, fontweight="bold",
            color="#F44336",
        )

        # Format x-axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30)

        plt.tight_layout()

        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {path}")
        else:
            plt.show()

        plt.close(fig)


def evaluate(
    ohlc: pd.DataFrame,
    strategies,
    capital: float = 1000.0,
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

    # ---- Validate position column ----------------------------------------
    if "position" not in signals.columns:
        raise ValueError("Strategy pipeline did not produce a 'position' column")

    positions = signals["position"].copy()

    # Replace NaN / inf with 0 (flat)
    bad_mask = positions.isna() | np.isinf(positions)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        print(f"  Warning: {n_bad} NaN/inf positions replaced with 0")
        positions = positions.fillna(0).replace([np.inf, -np.inf], 0)

    # Normalise to direction (sign) × scale (magnitude)
    # Standard positions are -1, 0, +1.  Scaled positions (e.g. from
    # DrawdownScale) can exceed ±1 — the magnitude acts as a leverage
    # multiplier on returns.
    max_abs = float(positions.abs().max())
    if max_abs > 1:
        print(f"  Note: scaled positions detected (max {max_abs:.2f}x)")

    signals["position"] = positions

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
    # Estimate bars per day from the index
    if len(equity_curve) > 1:
        total_seconds = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
        bar_seconds = total_seconds / (len(equity_curve) - 1)
        periods_per_day = 86400 / bar_seconds if bar_seconds > 0 else 1
    else:
        periods_per_day = 1
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
        prices=signals["close"],
        strategy_returns=strategy_returns,
    )


# ---------------------------------------------------------------------------
# Backtester – OO wrapper for streaming-style backtest
# ---------------------------------------------------------------------------


class Backtester:
    """Object-oriented backtester with a streaming candle-by-candle API.

    Usage
    -----
    ::

        bt = Backtester(cap=1000,
                        strategy=DrawdownPositionSize(signal=MACross(short=5, long=10), sharpe_window=1440, ...))

        for candle in candles:
            bt.ack(candle)
            bt.exec()

        bt.result("result.png")

    Parameters
    ----------
    cap : float
        Starting capital.
    strategy
        Strategy / risk-management overlay (must contain a signal and sharpe_window).
    """

    def __init__(self, cap: float, strategy) -> None:
        self.cap = cap
        self._strategy = strategy
        self._rows: list[dict] = []
        self._timestamps: list = []

    # -- streaming API ------------------------------------------------------

    def ack(self, candle) -> None:
        """Add a candle to the internal OHLCV history.

        *candle* can be a ``pd.Series`` whose ``.name`` is the timestamp,
        a ``dict`` with a ``"timestamp"`` key, or any object with
        ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``
        attributes.
        """
        if isinstance(candle, pd.Series):
            ts = candle.name
            row = {c: candle[c] for c in ("open", "high", "low", "close", "volume") if c in candle.index}
        elif isinstance(candle, dict):
            ts = candle.get("timestamp")
            row = {c: candle[c] for c in ("open", "high", "low", "close", "volume") if c in candle}
        else:
            ts = getattr(candle, "timestamp", None)
            row = {
                "open": getattr(candle, "open", None),
                "high": getattr(candle, "high", None),
                "low": getattr(candle, "low", None),
                "close": getattr(candle, "close", None),
                "volume": getattr(candle, "volume", None),
            }

        self._timestamps.append(pd.Timestamp(ts))
        self._rows.append({k: float(v) for k, v in row.items()})

    def exec(self) -> None:
        """Evaluate the latest candle (reserved for incremental logic)."""
        pass

    # -- results ------------------------------------------------------------

    def result(self, path: str | None = None) -> Result:
        """Run the full backtest on accumulated data and return metrics.

        Parameters
        ----------
        path : str, optional
            If given, save the equity/drawdown plot to this file path.
        """
        ohlc = pd.DataFrame(
            self._rows,
            index=pd.DatetimeIndex(self._timestamps, name="timestamp"),
        )

        strategies = [self._strategy]

        res = evaluate(ohlc, strategies, self.cap)
        res.sharpe_window = self._strategy.sharpe_window
        res.print()
        if path:
            res.plot(path)
        return res
