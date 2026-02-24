from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from src.app.config import UniverseConfig, ScreeningConfig
from src.app.logger import AppLogger
from src.okx.pool import OkxClientPool
from src.universe import Universe


class _Counter:
    def __init__(self, total: int):
        self.total = total
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


@dataclass
class TradeInstrument:
    inst_id: str
    daily_volume_usd: float
    trades: list[dict]


def _parse_bucket_interval(raw: str) -> int:
    """Parse e.g. '5m' into seconds."""
    raw = raw.strip().lower()
    if raw.endswith("m"):
        return int(raw[:-1]) * 60
    if raw.endswith("h"):
        return int(raw[:-1]) * 3600
    if raw.endswith("s"):
        return int(raw[:-1])
    return int(raw)


class UniverseFetcher:
    def __init__(self, pool: OkxClientPool, universe_cfg: UniverseConfig,
                 screening_cfg: ScreeningConfig, logger: AppLogger):
        self._pool = pool
        self._cfg = universe_cfg
        self._screening_cfg = screening_cfg
        self._logger = logger

    def fetch(self) -> list[TradeInstrument]:
        universe = Universe.discover(
            self._pool, self._cfg, self._logger,
            top_n=200,
        )
        self._logger.info(f"Universe after volume filter: {len(universe)}")

        return self._fetch_trades(universe)

    def _fetch_trades(self, universe: Universe) -> list[TradeInstrument]:
        lookback_ms = self._screening_cfg.lookback_hours * 3600 * 1000
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - lookback_ms

        specs = [
            (inst.inst_id, start_ms, now_ms)
            for inst in universe
        ]

        total = len(specs)
        self._logger.info(f"Fetching trades for {total} instruments concurrently")
        counter = _Counter(total)
        results: list[TradeInstrument] = self._pool.map(
            lambda pool, s: self._fetch_one(pool, s, counter),
            specs,
            max_workers=min(self._pool.pool_size, len(specs)),
        )
        return results

    def _fetch_one(
        self,
        pool: OkxClientPool,
        spec: tuple[str, int, int],
        counter: _Counter,
    ) -> TradeInstrument:
        inst_id, start_ms, end_ms = spec
        trades: list[dict] = []
        cursor = str(end_ms)

        while True:
            try:
                data = pool.public_get(
                    "/api/v5/market/history-trades",
                    params={
                        "instId": inst_id,
                        "limit": "100",
                        "after": cursor,
                        "type": "2",
                    },
                )
            except Exception as exc:
                self._logger.warning(
                    f"_fetch_one failed inst_id={inst_id} cursor={cursor} "
                    f"fetched_so_far={len(trades)}: {exc}",
                    exc_info=True,
                )
                break

            if not data:
                break

            trades.extend(data)
            oldest_ts = int(data[-1]["ts"])
            if oldest_ts <= start_ms or len(data) < 100:
                break
            if str(oldest_ts) == cursor:
                break
            cursor = str(oldest_ts)

        seen: set[str] = set()
        unique: list[dict] = []
        for t in trades:
            tid = t["tradeId"]
            if tid not in seen:
                seen.add(tid)
                if int(t["ts"]) >= start_ms:
                    unique.append(t)

        unique.sort(key=lambda t: int(t["ts"]))
        done = counter.increment()
        self._logger.info(f"[{done}/{counter.total}] {inst_id}: {len(unique)} trades")
        return TradeInstrument(inst_id=inst_id, daily_volume_usd=0.0, trades=unique)
