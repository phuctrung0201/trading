import uuid

from src.app.adapter import InfluxAdapter, SimulateAdapter
from src.app.core import CoreApp
from src.strategy.crossma import CrossMAStrategy
from src.strategy.drawdown import DrawdownStrategy


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
            app_config=self.config,
        )
        config = self.config.values.crossma
        self.logger.info(
            f"CrossMAStrategy configured short={config.short_length} long={config.long_length}"
        )
        return strategy

    def __init__(self, setup_name: str | None = None):
        super().__init__()
        self.setup_name = setup_name
        self.config = self.init_config(setup_name)
        self.logger = self.init_logger(self.config.values.log_level)
        self.logger.info(f"Initializing BacktestApp dependencies setup={setup_name or 'default'}")
        self.okx_client = self.init_okx_client(self.config)
        self.influx_client = (
            self.init_influxdb_client(self.config) if self.config.values.influx.enabled else None
        )
        self.logger.info(
            f"BacktestApp clients ready demo={self.config.values.trade.demo} "
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
        self.session_id = uuid.uuid4().hex
        self.measurement_adapter = InfluxAdapter(self.influx_client, session_id=self.session_id, setup_name=self.setup_name)
        self.logger.info(f"Backtest session_id={self.session_id}")
        self.warmup_periods = int(self.config.values.crossma.long_length)

        strategy_name = self.config.values.trade.strategy
        if strategy_name == "crossma":
            self.strategy = self.init_crossma_strategy(
                self.simulate_adapter,
                self.measurement_adapter,
            )
        else:
            self.strategy = self.init_drawdown_strategy(
                self.simulate_adapter,
                self.measurement_adapter,
            )
        self.logger.info(f"BacktestApp strategy={strategy_name}")
        self.logger.info("BacktestApp initialization completed")

    def init_drawdown_strategy(self, exchange_adapter, measurement_adapter):
        if self.config is None:
            raise RuntimeError("BacktestApp config must be initialized before strategies")
        if self.logger is None:
            raise RuntimeError("BacktestApp logger must be initialized before strategies")
        self.logger.info("Initializing DrawdownStrategy")
        strategy = DrawdownStrategy(
            exchange_adapter=exchange_adapter,
            measurement_adapter=measurement_adapter,
            app_logger=self.logger,
            app_config=self.config,
        )
        drawdown_config = self.config.values.drawdown
        self.logger.info(
            f"DrawdownStrategy configured window={drawdown_config.window} "
            f"thresholds={drawdown_config.threshold_scale_map}"
        )
        return strategy
