import argparse

from src.app.config import load_config, SetupConfig
from src.app.provider import AppProvider
from src.paper.app import PaperApp


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/paper/")
    return parser.parse_args()


def main():
    args = parse_args()
    provider = AppProvider()
    setup = load_config("paper", args.setup, SetupConfig)

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
            exchange=provider.okx_exchange,
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
            exchange=provider.okx_exchange,
            short_length=setup.crossma.short_length,
            long_length=setup.crossma.long_length,
            bucket_interval=setup.crossma.bucket_interval,
            window=setup.drawdown.window,
            threshold_scale_map=setup.drawdown.threshold_scale_map,
        )

    app = PaperApp(provider, strategy)
    logger = app.logger

    try:
        logger.info(f"Paper session_id {app.session_id}")

        total = 0
        try:
            for trade in app.exchange.stream_prices():
                total += 1
                try:
                    app.exchange.set_price(trade.price)
                    app.strategy.ack(trade)
                    app.emit_tick_ops()
                except Exception as exc:
                    logger.error(
                        f"Strategy execution failed timestamp={trade.timestamp} price={trade.price}: {exc}"
                    )
                    app.emit_error_ops(str(exc))
                    # Do not raise here to keep the paper trading process alive
        except KeyboardInterrupt:
            logger.info("Stopping paper trading...")
        logger.info(f"Paper completed total_trades={total}")
        logger.info(f"Paper session_id {app.session_id}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
