import logging

from src.app.trade import TradeApp


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

        logger.info(
            f"Starting live trade instrument={app.instrument} step={app.step}"
        )
        for candle in app.okx_client.stream_prices(
            instrument=app.instrument, step=app.step
        ):
            try:
                logger.info(
                    f"Candle timestamp={getattr(candle, 'timestamp', None)} "
                    f"close={getattr(candle, 'close', None)}"
                )
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
    finally:
        app.close()


if __name__ == "__main__":
    main()
