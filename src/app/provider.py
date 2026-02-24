import uuid

from src.app.config import (
    load_config,
    LoggerConfig,
    OkxConfig,
    ClickHouseConfig,
)
from src.app.logger import init_logger
from src.clickhouse.client import ClickHouseClient
from src.clickhouse.recorder import ClickHouseRecorder
from src.exchange.simulator import SimulateExchange
from src.okx.client import OkxClient
from src.okx.exchange import OkxExchange
from src.strategy.factory import StrategyFactory


class AppProvider:
    def __init__(self):
        app_cfg = load_config("app", "logger", LoggerConfig)
        okx_cfg = load_config("okx", "client", OkxConfig)
        ch_cfg = load_config("clickhouse", "client", ClickHouseConfig)

        self.logger = init_logger(app_cfg.log_level)
        self.session_id = uuid.uuid4().hex

        self.okx_client = OkxClient(
            api_key=okx_cfg.api_key,
            secret_key=okx_cfg.secret_key,
            passphrase=okx_cfg.passphrase,
            demo=okx_cfg.demo,
        )
        self.okx_exchange = OkxExchange(okx_client=self.okx_client, logger=self.logger)
        self.simulator = SimulateExchange()

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
                self.clickhouse_client = client
            except Exception as exc:
                self.logger.warning(
                    f"ClickHouse unavailable, recording disabled "
                    f"url={ch_cfg.url} database={ch_cfg.database}: {exc}",
                    exc_info=True,
                )

        self.recorder = ClickHouseRecorder(
            clickhouse_client=self.clickhouse_client,
            session_id=self.session_id,
        )

        self.strategy_factory = StrategyFactory(
            recorder=self.recorder,
            logger=self.logger,
        )
