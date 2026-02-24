from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.app.logger import AppLogger
from src.exchange.dto import FundingSnapshot
from src.instrument.repo import Instrument, InstrumentRepo, PricePair
from src.okx.pool import OkxClientPool
from src.trade.repo import TradeRepo

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

    def fetch_history(
        self, pair: str, perp_id: str, pool: OkxClientPool | None = None,
    ) -> Iterator[FundingSnapshot]:
        pool = pool or self._pool
        quote = perp_id.split("-")[1] if "-" in perp_id else "USDT"
        spot_id = f"{pair}-{quote}"

        data = pool.public_get(
            "/api/v5/public/funding-rate-history",
            params={"instId": perp_id, "limit": "100"},
        )
        if not data:
            self._logger.warning(f"No funding history for {perp_id}")
            return

        data.sort(key=lambda r: int(r.get("fundingTime", 0)))

        for entry in data:
            rate = float(entry.get("realizedRate", entry.get("fundingRate", "0")))
            ts = int(entry.get("fundingTime", 0))

            spot_price = self._fetch_candle_close(pool, spot_id, "4H", ts)
            perp_price = self._fetch_candle_close(pool, perp_id, "4H", ts)

            if spot_price <= 0 or perp_price <= 0:
                continue

            yield FundingSnapshot(
                timestamp=ts,
                funding_rate=rate,
                spot_price=spot_price,
                perp_price=perp_price,
                inst_id=perp_id,
            )

    def fetch_funding_history(
        self, instruments: list[Instrument],
    ) -> Iterator[FundingSnapshot]:
        self._logger.info(
            f"Fetching funding history for {len(instruments)} instruments"
        )

        results = self._pool.map(self._fetch_history_item, instruments)

        all_snapshots: list[FundingSnapshot] = []
        for snapshots in results:
            all_snapshots.extend(snapshots)

        all_snapshots.sort(key=lambda s: (s.timestamp, s.inst_id))
        yield from all_snapshots

    def _fetch_history_item(
        self, pool: OkxClientPool, inst: Instrument,
    ) -> list[FundingSnapshot]:
        snapshots = list(self.fetch_history(inst.pair, inst.inst_id, pool))
        self._logger.info(f"  {inst.inst_id}: {len(snapshots)} snapshots")
        return snapshots

    @staticmethod
    def _fetch_candle_close(
        pool: OkxClientPool, inst_id: str, bar: str, ts: int,
    ) -> float:
        params = {"instId": inst_id, "bar": bar, "after": str(ts + 1), "limit": "1"}
        try:
            candles = pool.public_get("/api/v5/market/candles", params=params)
            if candles:
                return float(candles[0][4])
            candles = pool.public_get(
                "/api/v5/market/history-candles", params=params,
            )
            if candles:
                return float(candles[0][4])
        except Exception:
            pass
        return 0.0
