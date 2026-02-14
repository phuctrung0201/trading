import logging

from src.app.backtest import BacktestApp


def main():
    try:
        run_backtest()
    except Exception as exc:
        logging.getLogger("trading.app").error(f"Backtest entry failed: {exc}")
        raise


def run_backtest():
    app = BacktestApp()
    if app.logger is None:
        raise RuntimeError("BacktestApp logger is not initialized")
    if app.okx_client is None:
        raise RuntimeError("BacktestApp okx_client is not initialized")
    logger = app.logger
    okx_client = app.okx_client

    try:
        try:
            candles = okx_client.get_prices(
                instrument=app.instrument,
                start=app.backtest_start,
                end=app.backtest_end,
                step=app.step,
            )
        except Exception:
            logger.error(
                "Failed to load historical candles "
                f"instrument={app.instrument} step={app.step} "
                f"start={app.backtest_start} end={app.backtest_end}"
            )
            logger.error("Historical candle fetch exception")
            raise
        logger.info(f"Loaded historical candles count={len(candles)}")
        if not candles:
            logger.warn("No candles returned for configured backtest window; strategy ack will not run")

        for candle in candles:
            try:
                app.crossma_strategy.ack(candle)
            except Exception:
                logger.error(
                    "Strategy ack failed "
                    f"timestamp={getattr(candle, 'timestamp', None)} "
                    f"close={getattr(candle, 'close', None)}"
                )
                logger.error("Strategy ack exception")
                raise
    finally:
        if app.influx_client is not None:
            app.influx_client.close()


if __name__ == "__main__":
    main()
