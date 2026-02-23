import argparse

from src.app.config import load_config, SetupConfig
from src.app.provider import AppProvider
from src.backtest.app import BacktestApp
from src.exchange.simulator import SimulateExchange


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/backtest/")
    return parser.parse_args()


def _build_trade_source(provider: AppProvider, setup):
    bt = setup.backtest
    dataset = bt.dataset
    if dataset:
        from src.clickhouse.dataset import DatasetReader
        if provider.clickhouse_client is None:
            raise RuntimeError("ClickHouse must be enabled for dataset backtest")
        reader = DatasetReader(provider.clickhouse_client, dataset, provider.logger)
        return reader.stream()
    depth_sec = bt.depth.total_minutes * 60 if bt.depth else None
    return provider.okx_exchange.stream_history(depth_sec=depth_sec)


def main():
    args = parse_args()
    provider = AppProvider()
    setup = load_config("backtest", args.setup, SetupConfig)

    bt = setup.backtest
    provider.simulator = SimulateExchange(
        initial_equity=bt.initial_equity,
        fee_rate=bt.fee_rate,
    )

    provider.okx_exchange.bootstrap(
        instrument=setup.instrument, leverage=setup.leverage,
    )
    provider.recorder.bootstrap(
        setup_name=args.setup, instrument=setup.instrument,
        strategy=setup.strategy,
    )

    if setup.strategy == "drawdown_meanrev":
        strategy = provider.drawdown_meanrev
        strategy.bootstrap(
            exchange=provider.simulator,
            bucket_interval=setup.meanrev.bucket_interval,
            window=setup.drawdown.window,
            threshold_scale_map=setup.drawdown.threshold_scale_map,
            lookback=setup.meanrev.lookback,
            entry_threshold=setup.meanrev.entry_threshold,
            exit_threshold=setup.meanrev.exit_threshold,
        )
    else:
        strategy = provider.drawdown_crossma
        strategy.bootstrap(
            exchange=provider.simulator,
            short_length=setup.crossma.short_length,
            long_length=setup.crossma.long_length,
            bucket_interval=setup.crossma.bucket_interval,
            window=setup.drawdown.window,
            threshold_scale_map=setup.drawdown.threshold_scale_map,
        )

    app = BacktestApp(provider, strategy)
    logger = app.logger

    try:
        total = 0
        for trade in _build_trade_source(provider, setup):
            total += 1
            try:
                app.exchange.set_price(trade.price)
                app.strategy.ack(trade)
            except Exception as exc:
                logger.error(
                    f"Strategy ack failed timestamp={trade.timestamp} "
                    f"price={trade.price} total_processed={total}: {exc}",
                    exc_info=True,
                )
                raise
        logger.info(f"Backtest completed total_trades={total}")
        if total == 0:
            logger.warning("No trades returned; strategy ack did not run")
        logger.info(f"Backtest session_id {app.session_id}")
        logger.info(f"Final equity={app.exchange.get_equity():.4f}")
        logger.info(f"Total fees={app.exchange.total_fees:.4f}")
    finally:
        if provider.clickhouse_client is not None:
            provider.clickhouse_client.close()


if __name__ == "__main__":
    main()
