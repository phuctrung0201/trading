from dataclasses import dataclass


@dataclass
class MarketTrade:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


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
