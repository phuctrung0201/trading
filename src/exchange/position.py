from src.exchange.types import Position


class PositionTracker:
    """In-memory position and equity tracking."""

    def __init__(self, initial_equity: float = 100.0):
        self.asset = initial_equity
        self.position: Position | None = None
        self._last_price: float = 0.0
        self._equity_at_open: float = 0.0

    def set_price(self, price: float):
        self._last_price = price

    def get_equity(self) -> float:
        return max(0.0, self.asset + self.unrealized_pnl())

    def unrealized_pnl(self) -> float:
        if self.position is None:
            return 0.0
        if self.position.price is None or self.position.price <= 0:
            return 0.0
        if self._last_price <= 0:
            return 0.0
        price_return = (self._last_price - self.position.price) / self.position.price
        direction = 1.0 if self.position.side == "buy" else -1.0
        return self.position.size * price_return * direction

    def open(self, position: Position) -> Position:
        self._equity_at_open = self.get_equity()
        fill_price = position.price if position.price is not None else self._last_price
        filled = Position(side=position.side, size=position.size, price=fill_price)
        self.position = filled
        return filled

    def close(self) -> float:
        pnl = self.unrealized_pnl()
        self.asset += pnl
        self.position = None
        self._equity_at_open = 0.0
        return pnl

    def get_asset(self, asset: str) -> float:
        _ = asset
        return self.asset

    def get_position(self) -> Position | None:
        return self.position
