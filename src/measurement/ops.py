from dataclasses import dataclass

from src.app.adapter import Measurable


@dataclass
class OpsMeasurement(Measurable):
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
