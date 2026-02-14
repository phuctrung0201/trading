"""Prometheus Pushgateway client helpers."""

from __future__ import annotations

import socket

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


class BacktestPrometheusClient:
    """Push backtest summary metrics to Pushgateway."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:9091",
        job: str = "trading_backtest",
        instance: str | None = None,
    ) -> None:
        self._gateway_url = gateway_url
        self._job = job
        self._instance = instance or socket.gethostname()
        self._registry = CollectorRegistry()

        self._starting_equity = Gauge(
            "trading_backtest_starting_equity",
            "Starting equity of the backtest run.",
            registry=self._registry,
        )
        self._final_equity = Gauge(
            "trading_backtest_final_equity",
            "Final equity of the backtest run.",
            registry=self._registry,
        )
        self._profit_pct = Gauge(
            "trading_backtest_profit_pct",
            "Backtest profit in percent.",
            registry=self._registry,
        )
        self._max_drawdown_pct = Gauge(
            "trading_backtest_max_drawdown_pct",
            "Maximum drawdown in percent (negative number).",
            registry=self._registry,
        )
        self._sharpe_ratio = Gauge(
            "trading_backtest_sharpe_ratio",
            "Backtest annualized Sharpe ratio.",
            registry=self._registry,
        )
        self._bars = Gauge(
            "trading_backtest_bars",
            "Number of bars used in the backtest.",
            registry=self._registry,
        )

    def push_metrics(
        self,
        starting_equity: float,
        final_equity: float,
        max_drawdown_pct: float,
        sharpe_ratio: float,
        bars: int,
        strategy: str = "unknown",
        instrument: str = "unknown",
        step: str = "unknown",
    ) -> None:
        """Set and push one backtest metric snapshot."""
        self._starting_equity.set(float(starting_equity))
        self._final_equity.set(float(final_equity))
        self._profit_pct.set((float(final_equity) / float(starting_equity) - 1.0) * 100.0)
        self._max_drawdown_pct.set(float(max_drawdown_pct))
        self._sharpe_ratio.set(float(sharpe_ratio))
        self._bars.set(int(bars))

        push_to_gateway(
            gateway=self._gateway_url,
            job=self._job,
            registry=self._registry,
            grouping_key={
                "instance": self._instance,
                "strategy": strategy,
                "instrument": instrument,
                "step": step,
            },
        )

