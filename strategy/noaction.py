from dataloader.ohlc import Candle
from strategy.action import Action, NoAction

# Base strategy
class NoActionStrategy:

    def __init__(self, equity: float) -> None:
        self._equity: float = equity

    # Pure analyze — decide the next action without mutating state
    def ack(self, candle: Candle) -> Action:
        return NoAction()

    # Update position tracking with the actual action
    def confirm(self, action: Action) -> None:
        pass

    def current_equity(self) -> float:
        return self._equity
