from abc import ABC, abstractmethod
from dataclasses import dataclass


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


class ExchangeAdapter(ABC):
    @abstractmethod
    def open(self, position: Position) -> OpenResult:
        raise NotImplementedError

    @abstractmethod
    def close(self, position: Position) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_asset(self, asset: str) -> float:
        raise NotImplementedError


class Measurable(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class MeasurementAdapter:
    @abstractmethod
    def record(self, measurable: Measurable):
        raise NotImplementedError
