import uuid

from src.app.core import CoreApp
from src.strategy.adapter import ExchangeAdapter, MeasurementAdapter, OpenResult, Position
from src.strategy.crossma import CrossMAStrategy


class SimulateAdapter(ExchangeAdapter):
    def __init__(self):
        self.asset = 100.0
        self.position: Position | None = None

    def open(self, position: Position) -> OpenResult:
        self.position = position
        return OpenResult(success=True, position=position, message=None)

    def close(self, position: Position) -> bool:
        if self.position is None:
            return False
        self.position = None
        return True

    def get_asset(self, asset: str) -> float:
        _ = asset
        return self.asset


class InfluxAdapter(MeasurementAdapter):
    def __init__(self, influxdb_client=None):
        self.influxdb_client = influxdb_client
        self.tags = self.init_tag_set()

    def new_session_id(self):
        return uuid.uuid4().hex

    def init_tag_set(self):
        return {"session_id": self.new_session_id()}

    def record(self, measurable):
        payload = {"tags": self.tags, "fields": measurable.to_dict()}
        timestamp = getattr(measurable, "timestamp", None)
        if timestamp is None:
            timestamp = payload["tags"]["session_id"]
        if self.influxdb_client is not None:
            self.influxdb_client.write(
                measurement="backtest",
                fields=payload["fields"],
                timestamp=timestamp,
            )
        return payload


class BacktestApp(CoreApp):
    def init_crossma_strategy(self, exchange_adapter, measurement_adapter):
        if self.config is None:
            raise RuntimeError("BacktestApp config must be initialized before strategies")
        if self.logger is None:
            raise RuntimeError("BacktestApp logger must be initialized before strategies")
        self.logger.info("Initializing CrossMAStrategy")
        strategy = CrossMAStrategy(
            exchange_adapter=exchange_adapter,
            measurement_adapter=measurement_adapter,
            app_logger=self.logger,
        )
        config = self.config.values.crossma
        strategy.init_short_length(config.short_length)
        strategy.init_long_length(config.long_length)
        self.logger.info(
            f"CrossMAStrategy configured short={config.short_length} long={config.long_length}"
        )
        return strategy

    def __init__(self):
        super().__init__()
        self.config = self.init_config()
        self.logger = self.init_logger(self.config.values.log_level)
        self.logger.info("Initializing BacktestApp dependencies")
        self.okx_client = self.init_okx_client(self.config)
        self.influx_client = (
            self.init_influxdb_client(self.config) if self.config.values.influx.enabled else None
        )
        self.logger.info(
            f"BacktestApp clients ready okx_demo={self.config.values.okx.demo} "
            f"influx_enabled={self.config.values.influx.enabled}"
        )

        self.instrument = self.config.values.trade.instrument
        self.step = self.config.values.trade.steps
        self.backtest_start = self.config.values.backtest.start
        self.backtest_end = self.config.values.backtest.end
        self.logger.info(
            f"BacktestApp config instrument={self.instrument} step={self.step} "
            f"start={self.backtest_start} end={self.backtest_end}"
        )

        self.simulate_adapter = SimulateAdapter()
        self.measurement_adapter = InfluxAdapter(self.influx_client)
        self.crossma_strategy = self.init_crossma_strategy(
            self.simulate_adapter,
            self.measurement_adapter,
        )
        self.logger.info("BacktestApp initialization completed")
