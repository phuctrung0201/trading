from dataclasses import dataclass


@dataclass
class MarketTrade:
    trade_id: str
    timestamp: str
    price: float
    size: float
    side: str


@dataclass
class Position:
    side: str
    size: float
    price: float | None = None


@dataclass
class OpenResult:
    success: bool
    position: Position | None = None
    message: str | None = None
