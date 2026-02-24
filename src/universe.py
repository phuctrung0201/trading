from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.app.config import UniverseConfig
from src.app.logger import AppLogger
from src.okx.pool import OkxClientPool


@dataclass(frozen=True)
class Instrument:
    pair: str
    inst_id: str


class Universe:
    """An immutable set of tradeable instruments discovered from the exchange.

    Shared by both the funding and portfolio pipelines.
    """

    def __init__(self, instruments: list[Instrument]):
        self._instruments = list(instruments)

    @classmethod
    def discover(
        cls,
        pool: OkxClientPool,
        config: UniverseConfig,
        logger: AppLogger,
        top_n: int = 10,
    ) -> Universe:
        """Query OKX for live perp instruments, filter by quote currency and
        24h volume, and return the top *top_n* by volume."""
        data = pool.public_get(
            "/api/v5/public/instruments",
            params={"instType": config.type},
        )

        tickers = pool.public_get(
            "/api/v5/market/tickers",
            params={"instType": config.type},
        )
        vol_map: dict[str, float] = {}
        for t in tickers:
            vol_map[t.get("instId", "")] = float(t.get("volCcy24h", "0"))

        instruments: list[Instrument] = []
        for inst in data:
            if inst.get("settleCcy", "") != config.quote:
                continue
            if inst.get("state", "") != "live":
                continue
            inst_id = inst["instId"]
            vol = vol_map.get(inst_id, 0.0)
            if vol < config.min_24h_volume_usd:
                continue
            pair = inst_id.replace(f"-{config.quote}-SWAP", "")
            instruments.append(Instrument(pair=pair, inst_id=inst_id))

        instruments.sort(key=lambda i: vol_map.get(i.inst_id, 0.0), reverse=True)
        result = cls(instruments[:top_n])
        logger.info(
            f"Universe discovered: {len(result)} instruments from {len(data)} total"
        )
        return result

    @property
    def instruments(self) -> list[Instrument]:
        return list(self._instruments)

    @property
    def inst_ids(self) -> list[str]:
        return [i.inst_id for i in self._instruments]

    def __len__(self) -> int:
        return len(self._instruments)

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._instruments)

    def __bool__(self) -> bool:
        return len(self._instruments) > 0

    def __repr__(self) -> str:
        ids = ", ".join(i.inst_id for i in self._instruments[:5])
        suffix = ", ..." if len(self._instruments) > 5 else ""
        return f"Universe([{ids}{suffix}], size={len(self._instruments)})"


class UniverseProvider:
    """Bound to a pool and logger so callers only pass the config."""

    def __init__(self, pool: OkxClientPool, logger: AppLogger):
        self._pool = pool
        self._logger = logger

    def discover(self, config: UniverseConfig, top_n: int = 10) -> Universe:
        return Universe.discover(self._pool, config, self._logger, top_n=top_n)
