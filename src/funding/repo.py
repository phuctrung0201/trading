from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.app.logger import AppLogger
from src.exchange.dto import FundingSnapshot
from src.instrument.repo import InstrumentRepo, PricePair
from src.okx.pool import OkxClientPool
from src.trade.repo import TradeRepo
from src.instrument.universe import Instrument

__all__ = ["FundingRate", "FundingRepo", "PricePair"]


@dataclass
class FundingRate:
    inst_id: str
    rate: float
    next_funding_time: int


class FundingRepo:
    def __init__(
        self,
        pool: OkxClientPool,
        instruments: InstrumentRepo,
        trades: TradeRepo,
        logger: AppLogger,
    ):
        self._pool = pool
        self._instruments = instruments
        self._trades = trades
        self._logger = logger

    def get_rate(self, inst_id: str) -> FundingRate:
        data = self._pool.public_get(
            "/api/v5/public/funding-rate",
            params={"instId": inst_id},
        )
        if not data:
            raise ValueError(f"No funding rate data for {inst_id}")
        entry = data[0]
        return FundingRate(
            inst_id=inst_id,
            rate=float(entry["fundingRate"]),
            next_funding_time=int(entry.get("nextFundingTime", 0)),
        )

    def get_rate_history(self, inst_id: str, limit: int = 100) -> list[dict]:
        data = self._pool.public_get(
            "/api/v5/public/funding-rate-history",
            params={"instId": inst_id, "limit": str(limit)},
        )
        return data or []

    def fetch_snapshot(self, perp_inst_id: str) -> FundingSnapshot:
        parts = perp_inst_id.split("-")
        spot_inst_id = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else perp_inst_id

        rate = self.get_rate(perp_inst_id)
        prices = self._instruments.get_price_pair(spot_inst_id, perp_inst_id)

        return FundingSnapshot(
            timestamp=rate.next_funding_time,
            funding_rate=rate.rate,
            spot_price=prices.spot_price,
            perp_price=prices.perp_price,
            inst_id=perp_inst_id,
        )

    def fetch_history(self, pair: str, perp_id: str) -> Iterator[FundingSnapshot]:
        quote = perp_id.split("-")[1] if "-" in perp_id else "USDT"
        spot_id = f"{pair}-{quote}"

        history = self.get_rate_history(perp_id, limit=100)
        if not history:
            self._logger.warning(f"No funding history for {perp_id}")
            return

        history.sort(key=lambda r: int(r.get("fundingTime", 0)))

        for entry in history:
            rate = float(entry.get("realizedRate", entry.get("fundingRate", "0")))
            ts = int(entry.get("fundingTime", 0))

            spot_price = self._trades.fetch_candle_close(spot_id, "4H", ts)
            perp_price = self._trades.fetch_candle_close(perp_id, "4H", ts)

            if spot_price <= 0 or perp_price <= 0:
                continue

            yield FundingSnapshot(
                timestamp=ts,
                funding_rate=rate,
                spot_price=spot_price,
                perp_price=perp_price,
                inst_id=perp_id,
            )

    def fetch_universe_history(
        self, instruments: list[Instrument],
    ) -> Iterator[FundingSnapshot]:
        self._logger.info(
            f"Fetching funding history for {len(instruments)} instruments"
        )

        all_snapshots: list[FundingSnapshot] = []
        for inst in instruments:
            snapshots = list(self.fetch_history(inst.pair, inst.inst_id))
            self._logger.info(f"  {inst.inst_id}: {len(snapshots)} snapshots")
            all_snapshots.extend(snapshots)

        all_snapshots.sort(key=lambda s: (s.timestamp, s.inst_id))
        yield from all_snapshots
