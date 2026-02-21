from datetime import datetime, timedelta, timezone

from src.app.provider import AppProvider
from src.clickhouse.measurement import OpsMeasurement
from src.drawdown.strategy import DrawdownStrategy


class PaperApp:
    def __init__(self, provider: AppProvider):
        self.logger = provider.logger
        self.exchange = provider.okx_exchange
        self.recorder = provider.recorder
        self.okx_client = provider.okx_client
        self.clickhouse_client = provider.clickhouse_client
        self.session_id = provider.session_id

        setup = provider.setup
        exchange_cfg = setup["exchange"]
        self.instrument = exchange_cfg["instrument"]
        self.step = exchange_cfg.get("steps", "1m")
        self.preload_duration = exchange_cfg.get("preload", "1d")

        crossma_cfg = setup.get("crossma", {})
        drawdown_cfg = setup.get("drawdown", {})
        self.strategy = DrawdownStrategy(
            exchange=self.exchange,
            recorder=self.recorder,
            logger=self.logger,
            short_length=int(crossma_cfg.get("short_length", 15)),
            long_length=int(crossma_cfg.get("long_length", 200)),
            steps=self.step,
            window=int(drawdown_cfg.get("window", 500)),
            threshold_scale_map=drawdown_cfg.get("threshold_scale_map", {0.0: 1.0}),
        )

        self.okx_client.set_api_callback(self._on_api_call)
        self.logger.info(
            f"PaperApp ready instrument={self.instrument} step={self.step}"
        )

    def _parse_duration(self, duration_str: str) -> timedelta:
        unit = duration_str[-1]
        count = int(duration_str[:-1]) if len(duration_str) > 1 else 1
        if unit == "m":
            return timedelta(minutes=count)
        if unit == "h":
            return timedelta(hours=count)
        if unit == "d":
            return timedelta(days=count)
        if unit == "w":
            return timedelta(weeks=count)
        return timedelta(days=1)

    def preload(self):
        duration = self._parse_duration(self.preload_duration)
        now = datetime.now(timezone.utc)
        start = (now - duration).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.logger.info(f"Preloading candles start={start} end={end} step={self.step}")

        total = 0
        for trade in self.okx_client.market.stream_history(
            instrument=self.instrument, start=start, end=end, step=self.step,
        ):
            total += 1
            if trade.close is not None:
                self.exchange.set_price(float(trade.close))
            self.strategy.warmup(trade)
        self.logger.info(f"Preload warm-up completed total={total}")

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
