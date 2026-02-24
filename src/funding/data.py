from __future__ import annotations

from dataclasses import dataclass

from src.exchange.dto import FundingSnapshot
from src.okx.pool import OkxClientPool


@dataclass
class FundingRate:
    inst_id: str
    rate: float
    next_funding_time: int


@dataclass
class PricePair:
    spot_price: float
    perp_price: float


def get_funding_rate(pool: OkxClientPool, inst_id: str) -> FundingRate:
    """Fetch current funding rate for a perp instrument.

    Calls OKX GET /api/v5/public/funding-rate (no auth).
    """
    data = pool.public_get(
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


def get_prices(pool: OkxClientPool, spot_inst_id: str, perp_inst_id: str) -> PricePair:
    """Fetch mark prices for a spot and perp instrument back-to-back."""
    spot_data = pool.public_get(
        "/api/v5/public/mark-price",
        params={"instId": spot_inst_id, "instType": "SPOT"},
    )
    perp_data = pool.public_get(
        "/api/v5/public/mark-price",
        params={"instId": perp_inst_id, "instType": "SWAP"},
    )

    if not spot_data:
        spot_data = pool.public_get(
            "/api/v5/market/ticker",
            params={"instId": spot_inst_id},
        )
        spot_price = float(spot_data[0]["last"]) if spot_data else 0.0
    else:
        spot_price = float(spot_data[0]["markPx"])

    if not perp_data:
        raise ValueError(f"No mark-price data for {perp_inst_id}")
    perp_price = float(perp_data[0]["markPx"])

    return PricePair(spot_price=spot_price, perp_price=perp_price)


def fetch_funding_snapshot(pool: OkxClientPool, perp_inst_id: str) -> FundingSnapshot:
    """Build a live FundingSnapshot from current OKX data."""
    parts = perp_inst_id.split("-")
    spot_inst_id = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else perp_inst_id

    rate = get_funding_rate(pool, perp_inst_id)
    prices = get_prices(pool, spot_inst_id, perp_inst_id)

    return FundingSnapshot(
        timestamp=rate.next_funding_time,
        funding_rate=rate.rate,
        spot_price=prices.spot_price,
        perp_price=prices.perp_price,
    )
