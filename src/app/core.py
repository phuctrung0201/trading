from src.app.config import AppConfig
from src.app.logger import AppLogger, init_logger as build_logger
from src.client.clickhouse import ClickHouseClient
from src.client.okx import OkxClient


class CoreApp:
    def init_config(self, setup_name: str):
        return AppConfig.load_setup(setup_name)

    def init_logger(self, log_level):
        return build_logger(log_level)

    def init_clickhouse_client(self, config):
        if self.logger is None:
            raise RuntimeError("CoreApp logger must be initialized before ClickHouseClient")
        ch = config.values.clickhouse
        client = ClickHouseClient(
            url=ch.url,
            database=ch.database,
            user=ch.user,
            password=ch.password,
            app_logger=self.logger,
        )
        client.ensure_tables()
        return client

    def init_okx_client(self, config):
        okx = config.value.okx
        return OkxClient(
            api_key=okx.api_key,
            secret_key=okx.secret_key,
            passphrase=okx.passphrase,
            demo=bool(config.value.trade.demo),
        )

    def __init__(self):
        self.config: AppConfig | None = None
        self.logger: AppLogger | None = None
        self.clickhouse_client: ClickHouseClient | None = None
        self.okx_client: OkxClient | None = None
