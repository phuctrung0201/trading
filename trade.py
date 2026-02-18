import logging
import sys

from src.app.trade import TradeApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


def main():
    try:
        run_trade()
    except Exception as exc:
        logging.getLogger("trading.app").error(f"Trade entry failed: {exc}")
        raise


def run_trade():
    app = TradeApp()
    if app.logger is None:
        raise RuntimeError("TradeApp logger is not initialized")
    if app.okx_client is None:
        raise RuntimeError("TradeApp okx_client is not initialized")
    logger = app.logger

    try:
        logger.info(f"Trade session_id={app.session_id}")
        app.preload()

        total = 0
        try:
            for candle in app.okx_client.stream_prices(
                instrument=app.instrument, step=app.step
            ):
                total += 1
                try:
                    close_value = getattr(candle, "close", None)
                    if close_value is not None:
                        app.exchange_adapter.set_price(float(close_value))
                    app.strategy.ack(candle)
                except Exception:
                    logger.error(
                        "Strategy ack failed "
                        f"timestamp={getattr(candle, 'timestamp', None)} "
                        f"close={getattr(candle, 'close', None)}"
                    )
                    logger.error("Strategy ack exception")
                    raise
        except KeyboardInterrupt:
            logger.info("Stopping live trading...")
        logger.info(f"Trade completed total_candles={total}")
        logger.info(f"Trade session_id={app.session_id}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
