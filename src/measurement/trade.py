from dataclasses import dataclass

from src.app.adapter import Measurable


@dataclass
class TradeMeasurement(Measurable):
    timestamp: int | None
    equity: float
    position_size: float
    position_side: str
    drawdown: float
    sharpe_ratio: float
    short_ema: float | None = None
    long_ema: float | None = None

    def to_dict(self) -> dict:
        result = {
            "equity": float(self.equity),
            "position_size": float(self.position_size),
            "position_side": str(self.position_side),
            "drawdown": float(self.drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
        }
        if self.short_ema is not None:
            result["short_ema"] = float(self.short_ema)
        if self.long_ema is not None:
            result["long_ema"] = float(self.long_ema)
        return result
