from __future__ import annotations

from dataclasses import dataclass

from src.okx.pool import OkxClientPool


@dataclass(frozen=True)
class Instrument:
    pair: str
    inst_id: str


@dataclass
class PricePair:
    spot_price: float
    perp_price: float


class InstrumentRepo:
    def __init__(self, pool: OkxClientPool):
        self._pool = pool

    def list_instruments(self, inst_type: str) -> list[dict]:
        return self._pool.public_get(
            "/api/v5/public/instruments",
            params={"instType": inst_type},
        )

    def get_instrument(self, inst_id: str, inst_type: str = "SWAP") -> dict:
        data = self._pool.public_get(
            "/api/v5/public/instruments",
            params={"instType": inst_type, "instId": inst_id},
        )
        if not data:
            raise ValueError(f"Instrument not found: {inst_id}")
        return data[0]

    def list_tickers(self, inst_type: str) -> list[dict]:
        return self._pool.public_get(
            "/api/v5/market/tickers",
            params={"instType": inst_type},
        )

    def get_mark_price(self, inst_id: str, inst_type: str = "SWAP") -> float:
        data = self._pool.public_get(
            "/api/v5/public/mark-price",
            params={"instType": inst_type, "instId": inst_id},
        )
        if not data:
            raise ValueError(f"No mark price for {inst_id}")
        return float(data[0]["markPx"])

    def get_price_pair(self, spot_id: str, perp_id: str) -> PricePair:
        """Falls back to ticker when SPOT mark-price is unavailable."""
        spot_data = self._pool.public_get(
            "/api/v5/public/mark-price",
            params={"instId": spot_id, "instType": "SPOT"},
        )
        perp_data = self._pool.public_get(
            "/api/v5/public/mark-price",
            params={"instId": perp_id, "instType": "SWAP"},
        )

        if not spot_data:
            spot_data = self._pool.public_get(
                "/api/v5/market/ticker",
                params={"instId": spot_id},
            )
            spot_price = float(spot_data[0]["last"]) if spot_data else 0.0
        else:
            spot_price = float(spot_data[0]["markPx"])

        if not perp_data:
            raise ValueError(f"No mark-price data for {perp_id}")
        perp_price = float(perp_data[0]["markPx"])

        return PricePair(spot_price=spot_price, perp_price=perp_price)
