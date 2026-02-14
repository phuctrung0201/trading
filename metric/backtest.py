"""Backtest metric with embedded registry snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from prometheus_client import CollectorRegistry, Gauge

from metric.collector import PushableMetric


@dataclass
class BacktestMetric(PushableMetric):
    JOB: ClassVar[str] = "backtest"

    strategy: str
    instrument: str
    step: str
    equity: float
    drawdown: float
    sharpe_ratio: float
    _registry: CollectorRegistry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._registry = CollectorRegistry()

        equity = Gauge(
            "backtest_equity",
            "Backtest equity.",
            registry=self._registry,
        )
        drawdown = Gauge(
            "backtest_drawdown",
            "Backtest drawdown.",
            registry=self._registry,
        )
        sharpe_ratio = Gauge(
            "backtest_sharpe_ratio",
            "Backtest annualized Sharpe ratio.",
            registry=self._registry,
        )

        equity.set(float(self.equity))
        drawdown.set(float(self.drawdown))
        sharpe_ratio.set(float(self.sharpe_ratio))

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

    @property
    def grouping_key(self) -> dict[str, str]:
        return {
            "strategy": str(self.strategy),
            "instrument": str(self.instrument),
            "step": str(self.step),
        }

    def get_job(self) -> str:
        return self.JOB

    def get_registry(self) -> CollectorRegistry:
        return self.registry

    def get_grouping_key(self) -> dict[str, str]:
        return self.grouping_key
