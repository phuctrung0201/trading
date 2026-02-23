from src.app.config import (
    load_config,
    LoggerConfig,
    OkxConfig,
    ClickHouseConfig,
    PortfolioConfig,
)
from src.app.logger import init_logger
from src.clickhouse.client import ClickHouseClient
from src.okx.client import OkxClient
from src.portfolio.pipeline import Pipeline


def main():
    app_cfg = load_config("app", "logger", LoggerConfig)
    logger = init_logger(app_cfg.log_level)

    okx_cfg = load_config("okx", "client", OkxConfig)
    okx_client = OkxClient(
        api_key=okx_cfg.api_key,
        secret_key=okx_cfg.secret_key,
        passphrase=okx_cfg.passphrase,
        demo=okx_cfg.demo,
    )

    ch_cfg = load_config("clickhouse", "client", ClickHouseConfig)
    ch_client = None
    if ch_cfg.enabled:
        try:
            ch_client = ClickHouseClient(
                url=ch_cfg.url,
                database=ch_cfg.database,
                user=ch_cfg.user,
                password=ch_cfg.password,
                app_logger=logger,
            )
            ch_client.ensure_tables()
        except Exception:
            logger.warning("ClickHouse unavailable, results will not be stored")

    portfolio_cfg = load_config("portfolio", "scanner", PortfolioConfig)

    pipeline = Pipeline(
        pool=okx_client.pool,
        config=portfolio_cfg,
        clickhouse=ch_client,
        logger=logger,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
