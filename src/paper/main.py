import argparse

from src.app.provider import AppProvider
from src.paper.app import PaperApp


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/paper/")
    return parser.parse_args()


def main():
    args = parse_args()
    provider = AppProvider(mode="paper", setup_name=args.setup)
    app = PaperApp(provider)
    logger = app.logger

    try:
        logger.info(f"Paper session_id {app.session_id}")
        app.preload()

        total = 0
        try:
            for trade in app.okx_client.market.stream_prices(
                instrument=app.instrument, step=app.step,
            ):
                total += 1
                try:
                    if trade.close is not None:
                        app.exchange.set_price(float(trade.close))
                    app.strategy.ack(trade)
                    app.emit_tick_ops()
                except Exception as exc:
                    logger.error(
                        f"Strategy ack failed timestamp={trade.timestamp} close={trade.close}"
                    )
                    app.emit_error_ops(str(exc))
                    raise
        except KeyboardInterrupt:
            logger.info("Stopping paper trading...")
        logger.info(f"Paper completed total_candles={total}")
        logger.info(f"Paper session_id {app.session_id}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
