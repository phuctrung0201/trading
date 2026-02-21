from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import Position, OpenResult
from src.exchange.position import PositionTracker


class SimulateExchange(ExchangeAdapter):
    """Backtesting: delegates everything to PositionTracker."""

    def __init__(self, initial_equity: float = 100.0):
        self._tracker = PositionTracker(initial_equity)

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

    def get_asset(self, asset: str) -> float:
        return self._tracker.get_asset(asset)

    def get_position(self) -> Position | None:
        return self._tracker.get_position()

    def get_equity(self) -> float:
        return self._tracker.get_equity()

    def unrealized_pnl(self) -> float:
        return self._tracker.unrealized_pnl()
