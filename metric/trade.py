"""Trade metric with embedded registry snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from prometheus_client import CollectorRegistry, Counter, Gauge

from metric.collector import PushableMetric


@dataclass
class TradeMetric(PushableMetric):
    JOB: ClassVar[str] = "trading_live"

    strategy: str
    instrument: str
    step: str
    price: float | None = None
    equity: float | None = None
    position_side: int | None = None
    position_size: float | None = None
    orders_inc: int = 0
    _registry: CollectorRegistry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._registry = CollectorRegistry()

        last_price = Gauge(
            "trading_live_last_price",
            "Most recent traded price.",
            registry=self._registry,
        )
        equity = Gauge(
            "trading_live_equity",
            "Current account equity.",
            registry=self._registry,
        )
        position_side = Gauge(
            "trading_live_position_side",
            "Current position side: -1 short, 0 flat, 1 long.",
            registry=self._registry,
        )
        position_size = Gauge(
            "trading_live_position_size",
            "Current normalized position size.",
            registry=self._registry,
        )
        orders_total = Counter(
            "trading_live_orders_total",
            "Total number of submitted orders.",
            registry=self._registry,
        )

        if self.price is not None:
            last_price.set(float(self.price))
        if self.equity is not None:
            equity.set(float(self.equity))
        if self.position_side is not None:
            position_side.set(int(self.position_side))
        if self.position_size is not None:
            position_size.set(float(self.position_size))
        if self.orders_inc > 0:
            orders_total.inc(int(self.orders_inc))

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
