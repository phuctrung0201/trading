import logging
import sys

from src.app.backtest import BacktestApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


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
        warmup_count = 0
        total = 0
        try:
            for candle in okx_client.stream_history(
                instrument=app.instrument,
                start=app.backtest_start,
                end=app.backtest_end,
                step=app.step,
            ):
                total += 1
                try:
                    close_value = getattr(candle, "close", None)
                    if close_value is not None:
                        app.simulate_adapter.set_price(float(close_value))
                    if warmup_count < app.warmup_periods:
                        app.strategy.warmup(candle)
                        warmup_count += 1
                    else:
                        app.strategy.ack(candle)
                except Exception:
                    logger.error(
                        "Strategy ack failed "
                        f"timestamp={getattr(candle, 'timestamp', None)} "
                        f"close={getattr(candle, 'close', None)}"
                    )
                    logger.error("Strategy ack exception")
                    raise
        except Exception:
            logger.error(
                "Failed during historical candle streaming "
                f"instrument={app.instrument} step={app.step} "
                f"start={app.backtest_start} end={app.backtest_end}"
            )
            raise
        logger.info(
            f"Backtest completed total_candles={total} "
            f"warmup={warmup_count} traded={total - warmup_count}"
        )
        if total == 0:
            logger.warn("No candles returned for configured backtest window; strategy ack did not run")
        logger.info(f"Backtest session_id={app.session_id}")
    finally:
        if app.influx_client is not None:
            app.influx_client.close()


if __name__ == "__main__":
    main()
