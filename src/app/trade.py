import uuid
from datetime import datetime, timedelta, timezone

from src.app.adapter import ClickHouseAdapter, OkxExchangeAdapter
from src.app.config import AppConfig
from src.app.core import CoreApp
from src.app.logger import AppLogger
from src.measurement.ops import OpsMeasurement
from src.strategy.crossma import CrossMAStrategy
from src.strategy.drawdown import DrawdownStrategy


class TradeApp(CoreApp):
    def __init__(self, setup_name: str):
        super().__init__()
        self.setup_name = setup_name
        self.config = self.init_config(setup_name)
        self.logger = self.init_logger(self.config.values.log_level)
        self.logger.info(f"Initializing TradeApp dependencies setup={setup_name or 'default'}")
        self.okx_client = self.init_okx_client(self.config)
        self.clickhouse_client = (
            self.init_clickhouse_client(self.config) if self.config.values.clickhouse.enabled else None
        )
        self.logger.info(
            f"TradeApp clients ready demo={self.config.values.trade.demo} "
            f"clickhouse_enabled={self.config.values.clickhouse.enabled}"
        )

        trade_config = self.config.values.trade
        self.instrument = trade_config.instrument
        self.step = trade_config.steps
        self.preload_duration = trade_config.preload
        self.strategy_name = trade_config.strategy
        self.logger.info(
            f"TradeApp config instrument={self.instrument} step={self.step} "
            f"preload={self.preload_duration}"
        )

        self.exchange_adapter = OkxExchangeAdapter(
            okx_client=self.okx_client,
            instrument=self.instrument,
            leverage=trade_config.leverage,
        )
        self.session_id = uuid.uuid4().hex

        self.measurement_adapter = ClickHouseAdapter(
            clickhouse_client=self.clickhouse_client,
            session_id=self.session_id,
            setup_name=self.setup_name,
            instrument=self.instrument,
            strategy=self.strategy_name,
        )

        self.okx_client.set_api_callback(self._on_api_call)

        if self.strategy_name == "crossma":
            self.strategy = self._init_crossma_strategy()
        else:
            self.strategy = self._init_drawdown_strategy()
        self.logger.info(f"TradeApp strategy={self.strategy_name}")
        self.logger.info("TradeApp initialization completed")

    def _init_crossma_strategy(self):
        assert self.logger is not None and self.config is not None
        self.logger.info("Initializing CrossMAStrategy")
        strategy = CrossMAStrategy(
            exchange_adapter=self.exchange_adapter,
            measurement_adapter=self.measurement_adapter,
            app_logger=self.logger,
            app_config=self.config,
        )
        config = self.config.values.crossma
        self.logger.info(
            f"CrossMAStrategy configured short={config.short_length} long={config.long_length}"
        )
        return strategy

    def _init_drawdown_strategy(self):
        assert self.logger is not None and self.config is not None
        self.logger.info("Initializing DrawdownStrategy")
        strategy = DrawdownStrategy(
            exchange_adapter=self.exchange_adapter,
            measurement_adapter=self.measurement_adapter,
            app_logger=self.logger,
            app_config=self.config,
        )
        drawdown_config = self.config.values.drawdown
        self.logger.info(
            f"DrawdownStrategy configured window={drawdown_config.window} "
            f"thresholds={drawdown_config.threshold_scale_map}"
        )
        return strategy

    def _parse_duration(self, duration_str):
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
        assert self.logger is not None and self.okx_client is not None
        duration = self._parse_duration(self.preload_duration)
        now = datetime.now(timezone.utc)
        start = (now - duration).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.logger.info(f"Preloading candles start={start} end={end} step={self.step}")

        total = 0
        for candle in self.okx_client.stream_history(
            instrument=self.instrument,
            start=start,
            end=end,
            step=self.step,
        ):
            total += 1
            close_value = getattr(candle, "close", None)
            if close_value is not None:
                self.exchange_adapter.set_price(float(close_value))
            self.strategy.warmup(candle)
        self.logger.info(f"Preload warm-up completed total={total}")

    def _on_api_call(self, latency_ms: int, response_code: int, source: str):
        ops = OpsMeasurement(
            timestamp=self._measurement_timestamp_ns(),
            type="api",
            api_latency_ms=latency_ms,
            response_code=response_code,
            response_source=source,
        )
        self.measurement_adapter.record(ops)

    def _measurement_timestamp_ns(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)

    def emit_tick_ops(self, candle_lag_ms: int | None = None):
        buf = self.clickhouse_client.buffer_size_total() if self.clickhouse_client else 0
        ops = OpsMeasurement(
            timestamp=self._measurement_timestamp_ns(),
            type="tick",
            candle_lag_ms=candle_lag_ms,
            write_buffer_size=buf,
        )
        self.measurement_adapter.record(ops)

    def emit_error_ops(self, message: str):
        ops = OpsMeasurement(
            timestamp=self._measurement_timestamp_ns(),
            type="error",
            error_message=message,
        )
        self.measurement_adapter.record(ops)

    def close(self):
        assert self.logger is not None
        self.logger.info("TradeApp closing")
        if self.clickhouse_client is not None:
            self.clickhouse_client.close()
        self.logger.info("TradeApp closed")
