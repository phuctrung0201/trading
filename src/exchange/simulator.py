from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import Position, OpenResult
from src.exchange.position import PositionTracker


class SimulateExchange(ExchangeAdapter):
    def __init__(self, initial_equity: float = 100.0, fee_rate: float = 0.0):
        self._tracker = PositionTracker(initial_equity, fee_rate=fee_rate)

    def set_price(self, price: float):
        self._tracker.set_price(price)

    def open(self, position: Position) -> OpenResult:
        filled = self._tracker.open(position)
        return OpenResult(success=True, position=filled)

    def close(self, position: Position) -> bool:
        if self._tracker.position is None:
            return False
        self._tracker.close()
        return True

    def adjust_equity(self, delta: float):
        self._tracker.adjust_equity(delta)

    def get_asset(self, asset: str) -> float:
        return self._tracker.get_asset(asset)

    def get_position(self) -> Position | None:
        return self._tracker.get_position()

    def get_equity(self) -> float:
        return self._tracker.get_equity()

    def unrealized_pnl(self) -> float:
        return self._tracker.unrealized_pnl()

    @property
    def total_fees(self) -> float:
        return self._tracker.total_fees

    @property
    def last_fee(self) -> float:
        return self._tracker.last_fee
