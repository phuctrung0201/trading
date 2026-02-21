from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.exchange.dto import MarketTrade, Position, OpenResult


class ExchangeAdapter(ABC):
    @abstractmethod
    def set_price(self, price: float):
        raise NotImplementedError

    @abstractmethod
    def open(self, position: Position) -> OpenResult:
        raise NotImplementedError

    @abstractmethod
    def close(self, position: Position) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_asset(self, asset: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_position(self) -> Position | None:
        raise NotImplementedError

    @abstractmethod
    def get_equity(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def unrealized_pnl(self) -> float:
        raise NotImplementedError

    def stream_history(self, depth_sec: int) -> Iterator[MarketTrade]:
        raise NotImplementedError

    def stream_prices(self) -> Iterator[MarketTrade]:
        raise NotImplementedError
