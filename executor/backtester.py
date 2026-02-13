"""Backtester executor — replays historical candles through a strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataloader.ohlc import Candle
from executor.noaction import NoActionExecution
from logger import log
from strategy.action import Open


class Backtester(NoActionExecution):
    """Streaming backtester that feeds candles to a strategy.

    On each candle the strategy is asked to analyse and return an action.
    If the action is :class:`Open`, the position's fill price is set to
    the candle's open price before confirming back to the strategy.

    After each confirm the strategy's equity is collected to track
    drawdown and Sharpe ratio over time.

    Usage
    -----
    ::

        bt = Backtester(strategy=DrawdownPositionSize(...))
        bt.run(ohlc)
        bt.summary()

    Parameters
    ----------
    strategy
        Strategy instance whose ``ack`` method returns an Action.
    """

    def __init__(self, strategy) -> None:
        self._strategy = strategy
        self._equity_history: list[tuple[str, float]] = []
        self._price_history: list[tuple[str, float]] = []

    def ack(self, candle: Candle) -> None:
        action = self._strategy.ack(candle)

        if isinstance(action, Open):
            action.position.price = candle.open

        self._strategy.confirm(action)
        self._equity_history.append((candle.timestamp, self._strategy.current_equity()))
        self._price_history.append((candle.timestamp, float(candle.close)))

    # -- metrics ------------------------------------------------------------

    @property
    def equity_curve(self) -> pd.Series:
        """Equity curve as a time-indexed Series."""
        timestamps, equities = (list(x) for x in zip(*self._equity_history)) if self._equity_history else ([], [])
        return pd.Series(
            equities,
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
            name="equity",
        )

    @property
    def prices(self) -> pd.Series:
        """Close prices as a time-indexed Series."""
        timestamps, closes = (list(x) for x in zip(*self._price_history)) if self._price_history else ([], [])
        return pd.Series(
            closes,
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
            name="close",
        )

    @property
    def returns(self) -> pd.Series:
        """Bar-to-bar returns derived from the equity curve."""
        return self.equity_curve.pct_change().fillna(0)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.05 = 5%)."""
        eq = self.equity_curve
        if len(eq) < 2:
            return 0.0
        peak = eq.expanding().max()
        dd = (eq - peak) / peak
        return float(dd.min())

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio over the full equity curve."""
        ret = self.returns
        if len(ret) < 2 or ret.std() == 0:
            return 0.0
        idx = pd.DatetimeIndex(ret.index)
        total_secs = (idx[-1] - idx[0]).total_seconds()
        bar_secs = total_secs / (len(ret) - 1)
        periods_per_day = 86400 / bar_secs if bar_secs > 0 else 1
        annualisation = np.sqrt(365 * periods_per_day)
        return float((ret.mean() / ret.std()) * annualisation)

    # -- summary ------------------------------------------------------------

    def summary(self) -> None:
        """Log a formatted summary of the backtest metrics."""
        eq = self.equity_curve
        if len(eq) < 2:
            log.info("Not enough data for a summary")
            return

        profit_pct = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        dd = self.max_drawdown * 100
        sharpe = self.sharpe_ratio

        log.info(
            f"Backtest Summary\n"
            f"  Equity:       {eq.iloc[0]:.4f} → {eq.iloc[-1]:.4f}  ({profit_pct:+.2f}%)\n"
            f"  Max Drawdown: {dd:+.2f}%\n"
            f"  Sharpe Ratio: {sharpe:.4f}\n"
            f"  Bars:         {len(eq)}"
        )

    # -- result -------------------------------------------------------------

    def result(self, path: str) -> None:
        """Save price, equity, Sharpe, and drawdown charts to *path*."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        eq = self.equity_curve
        if len(eq) < 2:
            return

        ret = self.returns
        running_max = eq.cummax()
        drawdown_pct = ((eq - running_max) / running_max) * 100

        # Rolling annualised Sharpe
        idx = pd.DatetimeIndex(ret.index)
        total_secs = (idx[-1] - idx[0]).total_seconds()
        bar_secs = total_secs / (len(ret) - 1) if len(ret) > 1 else 1
        periods_per_day = 86400 / bar_secs if bar_secs > 0 else 1
        annualisation = np.sqrt(365 * periods_per_day)
        win = min(int(periods_per_day * 30), len(ret))
        rolling_mean = ret.rolling(window=win, min_periods=win).mean()
        rolling_std = ret.rolling(window=win, min_periods=win).std()
        rolling_sharpe = ((rolling_mean / rolling_std) * annualisation).replace(
            [np.inf, -np.inf], np.nan
        )

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        fig.suptitle("Backtest Result", fontsize=14, fontweight="bold")

        # --- Equity curve ---
        ax1.plot(eq.index, eq, color="#2196F3", linewidth=1.2, label="Equity")
        ax1.fill_between(eq.index, eq.iloc[0], eq, alpha=0.08, color="#2196F3")
        ax1.axhline(eq.iloc[0], color="grey", linestyle="--", linewidth=0.8, label="Starting Capital")
        ax1.set_ylabel("Equity")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.annotate(
            f"Max: {eq.max():,.2f}",
            xy=(0.99, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold", color="#4CAF50",
        )
        ax1.annotate(
            f"Min: {eq.min():,.2f}",
            xy=(0.99, 0.89), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold", color="#F44336",
        )

        # --- Rolling Sharpe ratio ---
        ax2.plot(rolling_sharpe.index, rolling_sharpe, color="#FF9800", linewidth=1.0,
                 label=f"Rolling Sharpe ({win}-bar)")
        ax2.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        ax2.fill_between(
            rolling_sharpe.index, 0, rolling_sharpe,
            where=rolling_sharpe >= 0, alpha=0.15, color="#4CAF50",
        )
        ax2.fill_between(
            rolling_sharpe.index, 0, rolling_sharpe,
            where=rolling_sharpe < 0, alpha=0.15, color="#F44336",
        )
        sharpe_ann_color = "#4CAF50" if self.sharpe_ratio >= 0 else "#F44336"
        ax2.annotate(
            f"Annualized: {self.sharpe_ratio:.4f}",
            xy=(0.99, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10, fontweight="bold", color=sharpe_ann_color,
        )
        ax2.set_ylabel("Sharpe Ratio")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # --- Drawdown ---
        ax3.fill_between(drawdown_pct.index, 0, drawdown_pct, color="#F44336", alpha=0.35)
        ax3.plot(drawdown_pct.index, drawdown_pct, color="#F44336", linewidth=0.8, label="Drawdown")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Time")
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.annotate(
            f"Max DD: {self.max_drawdown * 100:+.2f}%",
            xy=(0.99, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9, fontweight="bold", color="#F44336",
        )

        # Format x-axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30)

        plt.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Chart saved: {path}")
