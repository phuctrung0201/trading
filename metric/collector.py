"""Metric collector that pushes Prometheus registry objects."""

from __future__ import annotations

from abc import ABC, abstractmethod

from prometheus_client import CollectorRegistry

from client.prometheus import PrometheusClient


class PushableMetric(ABC):
    """Metric contract accepted by MetricCollector.push()."""

    @abstractmethod
    def get_job(self) -> str:
        ...

    @abstractmethod
    def get_registry(self) -> CollectorRegistry:
        ...

    @abstractmethod
    def get_grouping_key(self) -> dict[str, str]:
        ...


class MetricCollector:
    """Collector that pushes registry instances to Pushgateway."""

    def __init__(self, prometheus_client: PrometheusClient) -> None:
        self._prometheus_client = prometheus_client

    def push(self, metric: PushableMetric) -> None:
        self._prometheus_client.push(
            job=metric.get_job(),
            registry=metric.get_registry(),
            grouping_key=metric.get_grouping_key(),
        )
