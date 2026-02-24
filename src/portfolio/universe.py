from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from src.app.config import UniverseConfig, ScreeningConfig
from src.app.logger import AppLogger
from src.instrument.repo import InstrumentRepo
from src.trade.repo import TradeRepo
from src.instrument.universe import Universe


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
    def __init__(
        self,
        instruments: InstrumentRepo,
        trades: TradeRepo,
        universe_cfg: UniverseConfig,
        screening_cfg: ScreeningConfig,
        logger: AppLogger,
    ):
        self._instruments = instruments
        self._trades = trades
        self._cfg = universe_cfg
        self._screening_cfg = screening_cfg
        self._logger = logger

    def fetch(self) -> list[TradeInstrument]:
        universe = Universe.discover(
            self._instruments, self._cfg, self._logger,
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
        raw_results = self._trades.fetch_ranges(specs)

        results: list[TradeInstrument] = []
        for i, trades in enumerate(raw_results):
            inst_id = specs[i][0]
            done = counter.increment()
            self._logger.info(f"[{done}/{counter.total}] {inst_id}: {len(trades)} trades")
            results.append(
                TradeInstrument(inst_id=inst_id, daily_volume_usd=0.0, trades=trades),
            )
        return results
