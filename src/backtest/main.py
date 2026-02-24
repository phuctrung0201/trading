import argparse

from src.app.config import load_config, StrategyConfig
from src.app.provider import AppProvider
from src.backtest.app import BacktestApp
from src.exchange.simulator import SimulateExchange
from src.instrument.universe import Universe


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/strategy/")
    return parser.parse_args()


def _build_trade_source(provider: AppProvider, setup: StrategyConfig):
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


def _run_funding(app: BacktestApp, provider: AppProvider, universe: Universe):
    total = 0
    for snapshot in provider.funding.fetch_universe_history(universe.instruments):
        total += 1
        try:
            app.strategy.ack_funding(snapshot)
        except Exception as exc:
            app.logger.error(
                f"Strategy ack_funding failed inst={snapshot.inst_id} "
                f"timestamp={snapshot.timestamp} "
                f"rate={snapshot.funding_rate} total_processed={total}: {exc}",
                exc_info=True,
            )
            raise
    return total


def _run_trade(app: BacktestApp, provider: AppProvider, setup: StrategyConfig):
    total = 0
    for trade in _build_trade_source(provider, setup):
        total += 1
        try:
            app.exchange.set_price(trade.price)
            app.strategy.ack_trade(trade)
        except Exception as exc:
            app.logger.error(
                f"Strategy ack_trade failed timestamp={trade.timestamp} "
                f"price={trade.price} total_processed={total}: {exc}",
                exc_info=True,
            )
            raise
    return total


def main():
    args = parse_args()
    provider = AppProvider()
    setup = load_config("strategy", args.setup, StrategyConfig)

    is_funding = setup.data_source == "funding"

    universe: Universe | None = None
    if setup.universe:
        universe = provider.universe.discover(setup.universe)

    provider.simulator = SimulateExchange(
        initial_equity=setup.funding.notional if is_funding else setup.backtest.initial_equity,
        fee_rate=0.0 if is_funding else setup.backtest.fee_rate,
    )

    instrument_label = ",".join(universe.inst_ids) if universe else setup.instrument
    provider.recorder.bootstrap(
        setup_name=args.setup, instrument=instrument_label,
        strategy=setup.strategy,
    )

    strategy = provider.strategy_factory.build(
        setup, exchange=provider.simulator, universe=universe,
    )
    app = BacktestApp(provider, strategy, provider.simulator)
    logger = app.logger

    try:
        if is_funding:
            total = _run_funding(app, provider, universe)
        else:
            total = _run_trade(app, provider, setup)

        logger.info(f"Backtest completed total_steps={total}")
        if total == 0:
            logger.warning("No data returned; strategy ack did not run")

        logger.info(f"Backtest session_id {app.session_id}")
        logger.info(f"Final equity={app.exchange.get_equity():.4f}")

        if is_funding:
            res = strategy.results()
            logger.info(f"  Funding earned={res.funding_earned:.4f}")
            logger.info(f"  Rebalances={res.rebalance_count}")
            logger.info(f"  Max drawdown={res.max_drawdown:.4%}")
            logger.info(f"  Positions opened={res.positions_opened} closed={res.positions_closed}")
        else:
            logger.info(f"Total fees={app.exchange.total_fees:.4f}")
    finally:
        if provider.clickhouse_client is not None:
            provider.clickhouse_client.close()


if __name__ == "__main__":
    main()
