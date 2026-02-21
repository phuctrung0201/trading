import uuid

from src.app.config import load_config
from src.app.logger import init_logger


class AppProvider:
    def __init__(self, mode: str, setup_name: str):
        app_cfg = load_config("app", "logger")
        okx_cfg = load_config("okx", "client")
        ch_cfg = load_config("clickhouse", "client")
        setup = load_config(mode, setup_name)

        self.logger = init_logger(app_cfg["log_level"])
        self.session_id = uuid.uuid4().hex
        self.setup = setup

        from src.okx.client import OkxClient
        from src.clickhouse.client import ClickHouseClient
        from src.clickhouse.recorder import ClickHouseRecorder
        from src.exchange.simulator import SimulateExchange
        from src.okx.exchange import OkxExchange

        self.okx_client = OkxClient(**okx_cfg)
        self.clickhouse_client = None
        if ch_cfg.get("enabled"):
            try:
                client = ClickHouseClient(
                    url=ch_cfg["url"],
                    database=ch_cfg["database"],
                    user=ch_cfg["user"],
                    password=ch_cfg["password"],
                    app_logger=self.logger,
                )
                client.ensure_tables()
                self.clickhouse_client = client
            except Exception:
                self.logger.warning("ClickHouse unavailable, recording disabled")

        self.simulator = SimulateExchange()
        self.okx_exchange = OkxExchange(
            okx_client=self.okx_client,
            instrument=setup["exchange"]["instrument"],
            leverage=setup["exchange"].get("leverage", 1),
        )

        self.recorder = ClickHouseRecorder(
            clickhouse_client=self.clickhouse_client,
            session_id=self.session_id,
            setup_name=setup_name,
            instrument=setup["exchange"]["instrument"],
            strategy=setup.get("strategy", "drawdown"),
        )
