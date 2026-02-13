"""Backtester executor — replays historical candles through a strategy."""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from client.influxdb import InfluxClient
from dataloader.ohlc import Candle
from executor.noaction import NoActionExecution
from logger import log
from monitor.measurement import BacktestMeasurement
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

    def __init__(
        self,
        strategy,
        influx_client: InfluxClient | None = None,
    ) -> None:
        self._strategy = strategy
        self._influx_client = influx_client
        self._backtest_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._backtest_measurement: BacktestMeasurement | None = None
        if self._influx_client is not None:
            self._backtest_measurement = BacktestMeasurement()
        self._equity_history: list[tuple[str, float]] = []
        self._drawdown_history: list[tuple[str, float]] = []
        self._sharpe_history: list[tuple[str, float]] = []
        self._peak_equity: float | None = None
        self._prev_equity: float | None = None
        self._first_ts_ns: int | None = None
        self._last_ts_ns: int | None = None
        self._return_count: int = 0
        self._return_mean: float = 0.0
        self._return_m2: float = 0.0

    def ack(self, candle: Candle) -> None:
        action = self._strategy.ack(candle)

        if isinstance(action, Open):
            action.position.price = candle.close

        self._strategy.confirm(action)
        equity = float(self._strategy.current_equity())
        timestamp_ns = self._timestamp_ns(candle)

        if self._first_ts_ns is None:
            self._first_ts_ns = timestamp_ns
        self._last_ts_ns = timestamp_ns

        if self._prev_equity is not None and self._prev_equity != 0:
            bar_ret = (equity - self._prev_equity) / self._prev_equity
            self._return_count += 1
            delta = bar_ret - self._return_mean
            self._return_mean += delta / self._return_count
            delta2 = bar_ret - self._return_mean
            self._return_m2 += delta * delta2
        self._prev_equity = equity

        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        peak = self._peak_equity if self._peak_equity is not None else equity
        drawdown = float((equity - peak) / peak) if peak > 0 else 0.0
        sharpe_ratio = self._current_sharpe_ratio()

        self._equity_history.append((candle.timestamp, equity))
        self._drawdown_history.append((candle.timestamp, drawdown))
        self._sharpe_history.append((candle.timestamp, sharpe_ratio))
        self._write_influx(
            timestamp_ns=timestamp_ns,
            equity=equity,
            drawdown=drawdown,
            sharpe_ratio=sharpe_ratio,
        )

    def close(self) -> None:
        if self._influx_client is not None:
            self._influx_client.close()

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
    def returns(self) -> pd.Series:
        """Bar-to-bar returns derived from the equity curve."""
        return self.equity_curve.pct_change().fillna(0)

    @property
    def drawdown_curve(self) -> pd.Series:
        """Per-candle drawdown history as a time-indexed Series."""
        timestamps, values = (list(x) for x in zip(*self._drawdown_history)) if self._drawdown_history else ([], [])
        return pd.Series(
            values,
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
            name="drawdown",
        )

    @property
    def sharpe_curve(self) -> pd.Series:
        """Per-candle annualized Sharpe history as a time-indexed Series."""
        timestamps, values = (list(x) for x in zip(*self._sharpe_history)) if self._sharpe_history else ([], [])
        return pd.Series(
            values,
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
            name="sharpe_ratio",
        )

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.05 = 5%)."""
        if not self._drawdown_history:
            return 0.0
        return float(min(dd for _, dd in self._drawdown_history))

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio over the full equity curve."""
        return self._current_sharpe_ratio()

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

    def _write_influx(
        self,
        *,
        timestamp_ns: int,
        equity: float,
        drawdown: float,
        sharpe_ratio: float,
    ) -> None:
        if self._influx_client is None or self._backtest_measurement is None:
            return
        value = self._backtest_measurement.values(
            backtest_id=self._backtest_id,
            timestamp_ns=timestamp_ns,
            drawdown=drawdown,
            equity=equity,
            sharpe_ratio=sharpe_ratio,
        )
        self._influx_client.write(value)

    def _timestamp_ns(self, candle: Candle) -> int:
        if candle.timestamp_ns is not None:
            return int(candle.timestamp_ns)
        return int(pd.to_datetime(candle.timestamp, utc=True).value)

    def _current_sharpe_ratio(self) -> float:
        if self._return_count < 2:
            return 0.0
        variance = self._return_m2 / (self._return_count - 1)
        if variance <= 0:
            return 0.0
        if self._first_ts_ns is None or self._last_ts_ns is None:
            return 0.0
        total_secs = (self._last_ts_ns - self._first_ts_ns) / 1_000_000_000
        if total_secs <= 0:
            return 0.0
        bar_secs = total_secs / self._return_count
        if bar_secs <= 0:
            return 0.0
        periods_per_day = 86400 / bar_secs
        annualisation = np.sqrt(365 * periods_per_day)
        std = np.sqrt(variance)
        return float((self._return_mean / std) * annualisation)

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

        sharpe_curve = self.sharpe_curve
        drawdown_pct = self.drawdown_curve * 100

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

        # --- Sharpe ratio ---
        ax2.plot(sharpe_curve.index, sharpe_curve, color="#FF9800", linewidth=1.0, label="Sharpe")
        ax2.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        ax2.fill_between(
            sharpe_curve.index, 0, sharpe_curve,
            where=sharpe_curve >= 0, alpha=0.15, color="#4CAF50",
        )
        ax2.fill_between(
            sharpe_curve.index, 0, sharpe_curve,
            where=sharpe_curve < 0, alpha=0.15, color="#F44336",
        )
        current_sharpe = float(sharpe_curve.iloc[-1]) if len(sharpe_curve) else 0.0
        sharpe_ann_color = "#4CAF50" if current_sharpe >= 0 else "#F44336"
        ax2.annotate(
            f"Annualized: {current_sharpe:.4f}",
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
            f"Max DD: {drawdown_pct.min():+.2f}%",
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
