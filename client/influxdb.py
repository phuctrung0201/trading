"""Minimal InfluxDB v2 line-protocol writer for backtest records."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, overload
from urllib.parse import quote

import requests

from logger import log
from monitor.measurement import BacktestMeasurement, MeasurementValue, NoValueMeasurement

@dataclass(slots=True, init=False)
class InfluxConfig:
    url: str
    org: str
    bucket: str
    token: str
    batch_size: int
    timeout_seconds: float
    flush_interval_seconds: float

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.org = "trading"
        self.bucket = "trading"
        self.token = token
        self.batch_size = 1000
        self.timeout_seconds = 10.0
        self.flush_interval_seconds = 5.0


class InfluxClient:
    """Buffered InfluxDB v2 writer with async flushing thread."""

    def __init__(self, config: InfluxConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._queue: queue.Queue[str] = queue.Queue()
        self._closed = threading.Event()
        self._flush_now = threading.Event()
        self._worker = threading.Thread(
            target=self._run_flush_worker,
            name="influx-writer",
            daemon=True,
        )
        self._worker.start()
        log.info(
            f"Influx client started: url={self._config.url} org={self._config.org} "
            f"bucket={self._config.bucket}"
        )
        log.debug(
            f"Influx client started: url={self._config.url} org={self._config.org} "
            f"bucket={self._config.bucket} batch_size={self._config.batch_size}"
        )

    @classmethod
    def from_config(cls, config: InfluxConfig) -> InfluxClient:
        return cls(config)

    def write(
        self,
        value: MeasurementValue,
    ) -> None:
        if self._closed.is_set():
            log.warn("Influx client is closed; dropping write")
            return
        self._queue.put(value.to_line())
        qsize = self._queue.qsize()
        if qsize == 1 or qsize % 1000 == 0:
            log.debug(f"Influx enqueue ok: queue_size={qsize}")

    def flush(self) -> None:
        if self._closed.is_set():
            return

        log.debug(f"Influx flush requested: queue_size={self._queue.qsize()}")
        self._flush_now.set()
        self._queue.join()
        while self._flush_now.is_set():
            time.sleep(0.01)
        log.debug("Influx flush completed")

    def _post_lines(self, lines: list[str]) -> None:
        if not lines:
            return

        payload = "\n".join(lines)
        url = (
            f"{self._config.url}/api/v2/write"
            f"?org={quote(self._config.org)}"
            f"&bucket={quote(self._config.bucket)}"
            f"&precision=ns"
        )
        try:
            response = self._session.post(
                url,
                data=payload,
                headers={
                    "Authorization": f"Token {self._config.token}",
                    "Content-Type": "text/plain; charset=utf-8",
                },
                timeout=self._config.timeout_seconds,
            )
            if response.status_code >= 300:
                log.warn(f"Influx write failed [{response.status_code}]: {response.text[:200]}")
            else:
                log.debug(
                    f"Influx write ok: status={response.status_code} "
                    f"batch_points={len(lines)}"
                )
        except requests.RequestException as exc:
            log.warn(f"Influx write exception: {exc}")

    def _run_flush_worker(self) -> None:
        batch: list[str] = []
        interval = max(float(self._config.flush_interval_seconds), 0.1)
        deadline = time.monotonic() + interval

        while not self._closed.is_set() or not self._queue.empty() or batch:
            timeout = max(deadline - time.monotonic(), 0.0)
            try:
                line = self._queue.get(timeout=timeout)
                batch.append(line)
            except queue.Empty:
                pass

            if not batch:
                if self._flush_now.is_set() and self._queue.empty():
                    self._flush_now.clear()
                deadline = time.monotonic() + interval
                continue

            should_flush = (
                len(batch) >= self._config.batch_size
                or self._flush_now.is_set()
                or time.monotonic() >= deadline
            )
            if should_flush:
                self._post_lines(batch)
                for _ in batch:
                    self._queue.task_done()
                batch = []
                deadline = time.monotonic() + interval
                if self._flush_now.is_set() and self._queue.empty():
                    self._flush_now.clear()

    def close(self) -> None:
        if self._closed.is_set():
            return

        log.info("Closing Influx client")
        self.flush()
        self._closed.set()
        self._worker.join(timeout=max(self._config.timeout_seconds, 1.0) + 1.0)
        if self._worker.is_alive():
            log.warn("Influx flush worker did not stop cleanly")
        self._session.close()
        log.info("Influx client closed")

    def __enter__(self) -> InfluxClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
