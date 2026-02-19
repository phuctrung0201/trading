from src.app.config import AppConfig
from src.app.logger import AppLogger, init_logger as build_logger
from src.client.influxdb import InfluxDBClient
from src.client.okx import OkxClient


class CoreApp:
    def init_config(self, setup_name: str | None = None):
        if setup_name:
            return AppConfig.load_setup(setup_name)
        return AppConfig.load_yaml()

    def init_logger(self, log_level):
        return build_logger(log_level)

    def init_influxdb_client(self, config):
        if self.logger is None:
            raise RuntimeError("CoreApp logger must be initialized before InfluxDBClient")
        influx = config.value.influx
        return InfluxDBClient(
            url=influx.url,
            token=influx.token,
            org=influx.org,
            bucket=influx.bucket,
            app_logger=self.logger,
        )

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
        self.influx_client: InfluxDBClient | None = None
        self.okx_client: OkxClient | None = None
