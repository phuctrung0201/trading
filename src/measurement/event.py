from dataclasses import dataclass

from src.app.adapter import Measurable


@dataclass
class TradeEventMeasurement(Measurable):
    timestamp: int | None
    event: str
    equity: float
    close_price: float | None
    position_size: float
    position_side: str
    drawdown: float
    sharpe_ratio: float
    short_ema: float | None = None
    long_ema: float | None = None
    exposure_ratio: float = 1.0
    fill_price: float | None = None
    pnl: float | None = None
    signal: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        result = {
            "event": self.event,
            "equity": float(self.equity),
            "position_size": float(self.position_size),
            "position_side": str(self.position_side),
            "drawdown": float(self.drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "exposure_ratio": float(self.exposure_ratio),
        }
        if self.close_price is not None:
            result["close_price"] = float(self.close_price)
        if self.short_ema is not None:
            result["short_ema"] = float(self.short_ema)
        if self.long_ema is not None:
            result["long_ema"] = float(self.long_ema)
        if self.fill_price is not None:
            result["fill_price"] = float(self.fill_price)
        if self.pnl is not None:
            result["pnl"] = float(self.pnl)
        if self.signal is not None:
            result["signal"] = self.signal
        if self.reason is not None:
            result["reason"] = self.reason
        return result
