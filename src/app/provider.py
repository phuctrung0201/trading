import uuid

from src.app.config import (
    load_config,
    LoggerConfig,
    OkxConfig,
    ClickHouseConfig,
    SetupConfig,
)
from src.app.logger import init_logger


class AppProvider:
    def __init__(self, mode: str, setup_name: str):
        app_cfg = load_config("app", "logger", LoggerConfig)
        okx_cfg = load_config("okx", "client", OkxConfig)
        ch_cfg = load_config("clickhouse", "client", ClickHouseConfig)
        setup = load_config(mode, setup_name, SetupConfig)

        self.logger = init_logger(app_cfg.log_level)
        self.session_id = uuid.uuid4().hex
        self.setup = setup

        from src.okx.client import OkxClient
        from src.clickhouse.client import ClickHouseClient
        from src.clickhouse.recorder import ClickHouseRecorder
        from src.exchange.simulator import SimulateExchange
        from src.okx.exchange import OkxExchange

        self.okx_client = OkxClient(
            api_key=okx_cfg.api_key,
            secret_key=okx_cfg.secret_key,
            passphrase=okx_cfg.passphrase,
            demo=okx_cfg.demo,
        )
        self.clickhouse_client = None
        if ch_cfg.enabled:
            try:
                client = ClickHouseClient(
                    url=ch_cfg.url,
                    database=ch_cfg.database,
                    user=ch_cfg.user,
                    password=ch_cfg.password,
                    app_logger=self.logger,
                )
                client.ensure_tables()
                self.clickhouse_client = client
            except Exception:
                self.logger.warning("ClickHouse unavailable, recording disabled")

        self.simulator = SimulateExchange()
        self.okx_exchange = OkxExchange(
            okx_client=self.okx_client,
            instrument=setup.instrument,
            leverage=setup.leverage,
        )
        self.exchange = self.okx_exchange if mode == "paper" else self.simulator

        self.recorder = ClickHouseRecorder(
            clickhouse_client=self.clickhouse_client,
            session_id=self.session_id,
            setup_name=setup_name,
            instrument=setup.instrument,
            strategy=setup.strategy,
            indicator="vwema",
        )

        from src.drawdown.strategy import DrawdownStrategy

        self.strategy = DrawdownStrategy(
            exchange=self.exchange,
            recorder=self.recorder,
            logger=self.logger,
            short_length=setup.crossma.short_length,
            long_length=setup.crossma.long_length,
            window=setup.drawdown.window,
            threshold_scale_map=setup.drawdown.threshold_scale_map,
        )
