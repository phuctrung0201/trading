from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TypeVar

from src.okx.auth import OkxAuth

T = TypeVar("T")
R = TypeVar("R")


class _TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Tokens refill continuously at *rate* per second up to *capacity*.
    ``acquire()`` blocks until a token is available, guaranteeing the
    long-run request rate never exceeds *rate* while still allowing
    short bursts up to *capacity*.
    """

    def __init__(self, rate: float, capacity: int | None = None):
        self._rate = rate
        self._capacity = float(capacity if capacity is not None else max(1, int(rate)))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


class OkxClientPool:
    """Pool of :class:`OkxAuth` clients for concurrent, rate-limited
    access to the OKX API.

    Each pooled client owns an independent ``requests.Session`` so TCP
    connections are reused per-client while multiple clients can have
    requests in-flight simultaneously.

    Throughput is governed by a shared token-bucket; concurrency is
    bounded by *pool_size* (the number of underlying clients).
    """

    POOL_SIZE = 4
    MAX_RPS = 10.0

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        demo: bool,
    ):
        self._pool_size = self.POOL_SIZE
        self._limiter = _TokenBucket(rate=self.MAX_RPS)
        self._queue: queue.SimpleQueue[OkxAuth] = queue.SimpleQueue()
        self._clients: list[OkxAuth] = []
        for _ in range(self.POOL_SIZE):
            client = OkxAuth(api_key, secret_key, passphrase, demo)
            self._clients.append(client)
            self._queue.put(client)

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def set_api_callback(self, callback):
        for client in self._clients:
            client.set_api_callback(callback)

    @contextmanager
    def acquire(self):
        """Check out a client, rate-limit, then yield it.

        The token-bucket ``acquire`` happens *after* we have a client so
        we don't consume a token while still waiting for a free slot.
        """
        client = self._queue.get()
        try:
            self._limiter.acquire()
            yield client
        finally:
            self._queue.put(client)

    def public_get(self, path: str, params: dict | None = None):
        with self.acquire() as client:
            return client.public_get(path, params)

    def signed_request(self, method: str, path: str, body=None):
        with self.acquire() as client:
            return client.signed_request(method, path, body)

    def map(
        self,
        fn: Callable[["OkxClientPool", T], R],
        items: Sequence[T],
        max_workers: int | None = None,
    ) -> list[R]:
        """Run *fn(pool, item)* concurrently for every item in *items*.

        *fn* receives the pool itself so it can make multiple
        rate-limited calls per item (e.g. paginated fetches).  Results
        are returned in the same order as *items*.
        """
        if not items:
            return []
        workers = min(max_workers or self._pool_size, len(items))
        results: list[R | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, self, item): i
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results  # type: ignore[return-value]
