from dataclasses import dataclass

from src.strategy.adapter import Measurable


@dataclass
class TradeMeasurement(Measurable):
    timestamp: int | None
    equity: float
    position_size: float
    position_side: str
    drawdown: float
    sharpe_ratio: float

    def to_dict(self) -> dict:
        return {
            "equity": float(self.equity),
            "position_size": float(self.position_size),
            "position_side": str(self.position_side),
            "drawdown": float(self.drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
        }
