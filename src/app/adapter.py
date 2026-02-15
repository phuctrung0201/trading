import uuid
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


class SimulateAdapter(ExchangeAdapter):
    def __init__(self):
        self.asset = 100.0
        self.position: Position | None = None

    def open(self, position: Position) -> OpenResult:
        self.position = position
        return OpenResult(success=True, position=position, message=None)

    def close(self, position: Position) -> bool:
        if self.position is None:
            return False
        self.position = None
        return True

    def get_asset(self, asset: str) -> float:
        _ = asset
        return self.asset


class OkxExchangeAdapter(ExchangeAdapter):
    def __init__(self, okx_client, instrument, leverage):
        self._okx = okx_client
        self._instrument = instrument
        self._leverage = leverage
        self._leverage_set = False

    def open(self, position: Position) -> OpenResult:
        if not self._leverage_set:
            self._okx.set_leverage(self._instrument, self._leverage)
            self._leverage_set = True

        side = position.side
        size = position.size
        try:
            data = self._okx.place_order(
                instrument=self._instrument,
                side=side,
                size=size,
            )
            fill_price = None
            if data and isinstance(data, list) and len(data) > 0:
                fill_price = data[0].get("avgPx")
                if fill_price is not None:
                    fill_price = float(fill_price)
            filled = Position(side=side, size=size, price=fill_price)
            return OpenResult(success=True, position=filled)
        except Exception as exc:
            return OpenResult(success=False, message=str(exc))

    def close(self, position: Position) -> bool:
        try:
            self._okx.close_position(
                instrument=self._instrument,
                side=position.side,
            )
            return True
        except Exception:
            return False

    def get_asset(self, asset: str) -> float:
        return self._okx.get_asset(currency=asset)


class InfluxAdapter(MeasurementAdapter):
    def __init__(self, influxdb_client, session_id):
        self.influxdb_client = influxdb_client
        self.tags = {"session_id": session_id}

    def record(self, measurable):
        payload = {"tags": self.tags, "fields": measurable.to_dict()}
        timestamp = getattr(measurable, "timestamp", None)
        if self.influxdb_client is not None:
            self.influxdb_client.write(
                measurement="backtest",
                fields=payload["fields"],
                timestamp=timestamp,
                tags=payload["tags"],
            )
        return payload
