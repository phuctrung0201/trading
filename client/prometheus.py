"""Prometheus Pushgateway client."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Timer

from prometheus_client import CollectorRegistry, push_to_gateway


@dataclass(frozen=True)
class _PushRequest:
    job: str
    registry: CollectorRegistry
    grouping_key: dict[str, str]


class PrometheusClient:
    """Pushgateway client with optional delayed batching."""

    _FLUSH_DELAY_SECONDS = 3.0
    _BATCH_SIZE = 100

    def __init__(self, gateway_url: str) -> None:
        self.gateway_url = gateway_url
        self._flush_delay_seconds = float(self._FLUSH_DELAY_SECONDS)
        self._batch_size = int(self._BATCH_SIZE)
        self._lock = Lock()
        self._pending: dict[tuple[str, tuple[tuple[str, str], ...]], _PushRequest] = {}
        self._flush_timer: Timer | None = None

    def push(
        self,
        *,
        job: str,
        registry: CollectorRegistry,
        grouping_key: dict[str, str] | None = None,
    ) -> None:
        grouping = grouping_key or {}
        if self._flush_delay_seconds <= 0:
            self._send(job=job, registry=registry, grouping_key=grouping)
            return

        key = self._request_key(job=job, grouping_key=grouping)
        should_flush_now = False
        with self._lock:
            self._pending[key] = _PushRequest(
                job=job,
                registry=registry,
                grouping_key=grouping,
            )
            if len(self._pending) >= self._batch_size:
                should_flush_now = True
            if self._flush_timer is None:
                self._flush_timer = Timer(self._flush_delay_seconds, self.flush)
                self._flush_timer.daemon = True
                self._flush_timer.start()
        if should_flush_now:
            self.flush()

    def flush(self) -> None:
        """Flush pending batched requests immediately."""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
            self._flush_timer = None

        for req in pending:
            self._send(
                job=req.job,
                registry=req.registry,
                grouping_key=req.grouping_key,
            )

    def close(self) -> None:
        """Flush any pending pushes and stop timer."""
        timer: Timer | None = None
        with self._lock:
            timer = self._flush_timer
            self._flush_timer = None
        if timer is not None:
            timer.cancel()
        self.flush()

    @staticmethod
    def _request_key(
        *,
        job: str,
        grouping_key: dict[str, str],
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        # Same job + same labels are coalesced to latest request in a batch window.
        return job, tuple(sorted((str(k), str(v)) for k, v in grouping_key.items()))

    def _send(
        self,
        *,
        job: str,
        registry: CollectorRegistry,
        grouping_key: dict[str, str],
    ) -> None:
        push_to_gateway(
            gateway=self.gateway_url,
            job=str(job),
            registry=registry,
            grouping_key={str(k): str(v) for k, v in grouping_key.items()},
        )
