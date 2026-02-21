from abc import ABC, abstractmethod

from src.exchange.types import Position, OpenResult


class Exchange(ABC):
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
