import argparse
import time

from src.app.config import load_config, StrategyConfig
from src.app.provider import AppProvider
from src.exchange.simulator import SimulateExchange
from src.instrument.universe import Universe
from src.paper.app import PaperApp

_FUNDING_POLL_SEC = 60


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/strategy/")
    return parser.parse_args()


def _run_funding(app: PaperApp, provider: AppProvider, universe: Universe):
    logger = app.logger
    last_ts: dict[str, int | None] = {i.inst_id: None for i in universe}
    total = 0

    while True:
        for inst in universe:
            inst_id = inst.inst_id
            try:
                snapshot = provider.funding.fetch_snapshot(inst_id)
            except Exception as exc:
                logger.error(f"Funding fetch failed for {inst_id}: {exc}", exc_info=True)
                app.emit_error_ops(str(exc))
                continue

            if snapshot.timestamp == last_ts[inst_id]:
                continue

            last_ts[inst_id] = snapshot.timestamp
            total += 1
            try:
                app.strategy.ack_funding(snapshot)
                app.emit_tick_ops()
            except Exception as exc:
                logger.error(
                    f"Strategy ack_funding failed inst={inst_id} "
                    f"timestamp={snapshot.timestamp} "
                    f"rate={snapshot.funding_rate} total_processed={total}: {exc}",
                    exc_info=True,
                )
                app.emit_error_ops(str(exc))

        time.sleep(_FUNDING_POLL_SEC)


def _run_trade(app: PaperApp):
    total = 0

    for trade in app.exchange.stream_prices():
        total += 1
        try:
            app.exchange.set_price(trade.price)
            app.strategy.ack_trade(trade)
            app.emit_tick_ops()
        except Exception as exc:
            app.logger.error(
                f"Strategy execution failed timestamp={trade.timestamp} "
                f"price={trade.price} total_processed={total}: {exc}",
                exc_info=True,
            )
            app.emit_error_ops(str(exc))

    return total


def main():
    args = parse_args()
    provider = AppProvider()
    setup = load_config("strategy", args.setup, StrategyConfig)

    is_funding = setup.data_source == "funding"

    universe: Universe | None = None
    if setup.universe:
        universe = provider.universe.discover(setup.universe)

    if is_funding:
        provider.simulator = SimulateExchange(
            initial_equity=setup.funding.notional,
            fee_rate=0.0,
        )
        exchange = provider.simulator
    else:
        provider.okx_exchange.bootstrap(
            instrument=setup.instrument, leverage=setup.leverage,
        )
        exchange = provider.okx_exchange

    instrument_label = ",".join(universe.inst_ids) if universe else setup.instrument
    provider.recorder.bootstrap(
        setup_name=args.setup, instrument=instrument_label,
        strategy=setup.strategy,
    )

    strategy = provider.strategy_factory.build(
        setup, exchange=exchange, universe=universe,
    )
    app = PaperApp(provider, strategy, exchange)
    logger = app.logger

    try:
        logger.info(f"Paper session_id {app.session_id}")

        try:
            if is_funding:
                _run_funding(app, provider, universe)
            else:
                total = _run_trade(app)
                logger.info(f"Paper completed total_trades={total}")
        except KeyboardInterrupt:
            logger.info("Stopping paper trading...")

        logger.info(f"Paper session_id {app.session_id}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
