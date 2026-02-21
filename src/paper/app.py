from datetime import datetime, timezone

from src.app.provider import AppProvider
from src.clickhouse.measurement import OpsMeasurement


class PaperApp:
    def __init__(self, provider: AppProvider):
        self.logger = provider.logger
        self.exchange = provider.exchange
        self.strategy = provider.strategy
        self.recorder = provider.recorder
        self.clickhouse_client = provider.clickhouse_client
        self.session_id = provider.session_id

        provider.okx_client.set_api_callback(self._on_api_call)
        self.logger.info(f"PaperApp ready instrument={provider.setup.instrument}")

    def _on_api_call(self, latency_ms: int, response_code: int, source: str):
        ops = OpsMeasurement(
            timestamp=self._now_ns(),
            type="api",
            api_latency_ms=latency_ms,
            response_code=response_code,
            response_source=source,
        )
        self.recorder.record(ops)

    def _now_ns(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)

    def emit_tick_ops(self, candle_lag_ms: int | None = None):
        buf = self.clickhouse_client.buffer_size_total() if self.clickhouse_client else 0
        ops = OpsMeasurement(
            timestamp=self._now_ns(),
            type="tick",
            candle_lag_ms=candle_lag_ms,
            write_buffer_size=buf,
        )
        self.recorder.record(ops)

    def emit_error_ops(self, message: str):
        ops = OpsMeasurement(
            timestamp=self._now_ns(),
            type="error",
            error_message=message,
        )
        self.recorder.record(ops)

    def close(self):
        self.logger.info("PaperApp closing")
        if self.clickhouse_client is not None:
            self.clickhouse_client.close()
        self.logger.info("PaperApp closed")
