from __future__ import annotations

from collections.abc import Iterator

from src.app.config import MinFrTotalConfig, UniverseConfig
from src.app.logger import AppLogger
from src.funding.repo import FundingRepo
from src.instrument.repo import Instrument, InstrumentRepo


class Universe:
    """An immutable set of tradeable instruments discovered from the exchange.

    Shared by both the funding and portfolio pipelines.
    """

    def __init__(self, instruments: list[Instrument]):
        self._instruments = list(instruments)

    @classmethod
    def discover(
        cls,
        repo: InstrumentRepo,
        config: UniverseConfig,
        logger: AppLogger,
        funding: FundingRepo | None = None,
    ) -> Universe:
        """Query OKX for live perp instruments, filter by quote currency,
        24h volume, and optionally by average funding rate."""
        data = repo.list_instruments(config.type)
        tickers = repo.list_tickers(config.type)

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
            if vol < config.min_vol_usd.value:
                continue
            pair = inst_id.replace(f"-{config.quote}-SWAP", "")
            instruments.append(Instrument(pair=pair, inst_id=inst_id))

        instruments.sort(key=lambda i: vol_map.get(i.inst_id, 0.0), reverse=True)

        if config.min_fr_total and funding:
            instruments = _filter_by_funding_total(
                instruments, funding, config.min_fr_total, logger,
            )

        result = cls(instruments)
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


def _filter_by_funding_total(
    instruments: list[Instrument],
    funding: FundingRepo,
    cfg: MinFrTotalConfig,
    logger: AppLogger,
) -> list[Instrument]:
    result: list[Instrument] = []
    for inst in instruments:
        periods = cfg.depth_days * 3
        history = funding.get_rate_history(inst.inst_id, limit=periods)
        if not history:
            logger.info(f"  {inst.inst_id}: no funding history, skipped")
            continue
        rates = [
            float(e.get("realizedRate", e.get("fundingRate", "0")))
            for e in history
        ]
        total = sum(abs(r) for r in rates)
        if total < cfg.value:
            logger.info(
                f"  {inst.inst_id}: total_fr={total:.6f} < {cfg.value}, skipped"
            )
            continue
        logger.info(f"  {inst.inst_id}: total_fr={total:.6f}, kept")
        result.append(inst)
    return result


class UniverseProvider:
    def __init__(
        self,
        instruments: InstrumentRepo,
        logger: AppLogger,
        funding: FundingRepo | None = None,
    ):
        self._instruments = instruments
        self._logger = logger
        self._funding = funding

    def discover(self, config: UniverseConfig) -> Universe:
        return Universe.discover(
            self._instruments, config, self._logger, funding=self._funding,
        )
