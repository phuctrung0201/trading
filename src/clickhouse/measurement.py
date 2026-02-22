from dataclasses import dataclass


@dataclass
class TradeMeasurement:
    timestamp: int | None
    equity: float
    position_size: float
    position_side: str
    drawdown: float
    sharpe_ratio: float
    short_ema: float | None = None
    long_ema: float | None = None
    close_price: float | None = None
    exposure_ratio: float = 1.0
    zscore: float | None = None

    def to_dict(self) -> dict:
        result = {
            "equity": float(self.equity),
            "position_size": float(self.position_size),
            "position_side": str(self.position_side),
            "drawdown": float(self.drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "exposure_ratio": float(self.exposure_ratio),
        }
        if self.short_ema is not None:
            result["short_ema"] = float(self.short_ema)
        if self.long_ema is not None:
            result["long_ema"] = float(self.long_ema)
        if self.close_price is not None:
            result["close_price"] = float(self.close_price)
        if self.zscore is not None:
            result["zscore"] = float(self.zscore)
        return result


@dataclass
class TradeEventMeasurement:
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
    zscore: float | None = None
    fee: float | None = None

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
        if self.zscore is not None:
            result["zscore"] = float(self.zscore)
        if self.fee is not None:
            result["fee"] = float(self.fee)
        return result


@dataclass
class OpsMeasurement:
    timestamp: int | None
    type: str
    candle_lag_ms: int | None = None
    write_buffer_size: int | None = None
    api_latency_ms: int | None = None
    response_code: int | None = None
    response_source: str | None = None
    reconcile_equity_diff: float | None = None
    reconcile_position_match: bool | None = None
    reconcile_correction: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        result: dict = {"type": self.type}
        if self.candle_lag_ms is not None:
            result["candle_lag_ms"] = self.candle_lag_ms
        if self.write_buffer_size is not None:
            result["write_buffer_size"] = self.write_buffer_size
        if self.api_latency_ms is not None:
            result["api_latency_ms"] = self.api_latency_ms
        if self.response_code is not None:
            result["response_code"] = self.response_code
        if self.response_source is not None:
            result["response_source"] = self.response_source
        if self.reconcile_equity_diff is not None:
            result["reconcile_equity_diff"] = float(self.reconcile_equity_diff)
        if self.reconcile_position_match is not None:
            result["reconcile_position_match"] = self.reconcile_position_match
        if self.reconcile_correction is not None:
            result["reconcile_correction"] = self.reconcile_correction
        if self.error_message is not None:
            result["error_message"] = self.error_message
        return result
