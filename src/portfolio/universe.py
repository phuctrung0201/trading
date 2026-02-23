from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from src.app.config import UniverseConfig, ScreeningConfig
from src.app.logger import AppLogger
from src.okx.pool import OkxClientPool


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
class Instrument:
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

    def fetch(self) -> list[Instrument]:
        raw = self._discover()
        self._logger.info(f"Universe discovered {len(raw)} instruments")

        filtered = self._filter_volume(raw)
        self._logger.info(f"Universe after volume filter: {len(filtered)}")

        return self._fetch_trades(filtered)

    def _discover(self) -> list[dict]:
        data = self._pool.public_get(
            "/api/v5/public/instruments",
            params={"instType": self._cfg.type},
        )
        return [
            d for d in data
            if d.get("settleCcy", "") == self._cfg.quote
            and d.get("state", "") == "live"
        ]

    def _filter_volume(self, raw_instruments: list[dict]) -> list[tuple[str, float]]:
        tickers = self._pool.public_get(
            "/api/v5/market/tickers",
            params={"instType": self._cfg.type},
        )
        vol_map: dict[str, float] = {}
        for t in tickers:
            inst_id = t.get("instId", "")
            vol_usd = float(t.get("volCcy24h", "0"))
            vol_map[inst_id] = vol_usd

        result: list[tuple[str, float]] = []
        for inst in raw_instruments:
            inst_id = inst["instId"]
            vol = vol_map.get(inst_id, 0.0)
            if vol >= self._cfg.min_24h_volume_usd:
                result.append((inst_id, vol))
        return result

    def _fetch_trades(self, instruments: list[tuple[str, float]]) -> list[Instrument]:
        lookback_ms = self._screening_cfg.lookback_hours * 3600 * 1000
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - lookback_ms

        specs = [
            (inst_id, vol, start_ms, now_ms)
            for inst_id, vol in instruments
        ]

        total = len(specs)
        self._logger.info(f"Fetching trades for {total} instruments concurrently")
        counter = _Counter(total)
        results: list[Instrument] = self._pool.map(
            lambda pool, s: self._fetch_one(pool, s, counter),
            specs,
            max_workers=min(self._pool.pool_size, len(specs)),
        )
        return results

    def _fetch_one(
        self,
        pool: OkxClientPool,
        spec: tuple[str, float, int, int],
        counter: _Counter,
    ) -> Instrument:
        inst_id, vol, start_ms, end_ms = spec
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
                    f"Fetch failed {inst_id} cursor={cursor}: {exc}"
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
        return Instrument(inst_id=inst_id, daily_volume_usd=vol, trades=unique)
