import csv
import logging
import os
import sys

from src.app.backtest import BacktestApp
from src.client.ohclv import OHCLV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

DATA_DIR = "data"
CANDLE_FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


def _data_path(instrument: str, step: str, start: str, end: str) -> str:
    safe = lambda s: str(s).replace("/", "_").replace(":", "-").replace(" ", "_")
    filename = f"{safe(instrument)}_{step}_{safe(start)}_{safe(end)}.csv"
    return os.path.join(DATA_DIR, filename)


def download_history(app, path: str):
    """Stream candles from OKX and persist them to a CSV file."""
    logger = app.logger
    okx_client = app.okx_client
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger.info(f"Downloading history to {path}")
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDLE_FIELDS)
        writer.writeheader()
        for candle in okx_client.stream_history(
            instrument=app.instrument,
            start=app.backtest_start,
            end=app.backtest_end,
            step=app.step,
        ):
            writer.writerow({
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            })
            count += 1
    logger.info(f"Downloaded {count} candles to {path}")


def load_candles(path: str):
    """Yield OHCLV objects from a previously downloaded CSV file."""
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield OHCLV(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
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

    data_file = _data_path(app.instrument, app.step, app.backtest_start, app.backtest_end)

    if not os.path.exists(data_file):
        download_history(app, data_file)
    else:
        logger.info(f"Using cached history file {data_file}")

    try:
        warmup_count = 0
        total = 0
        try:
            for candle in load_candles(data_file):
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
                "Failed during candle replay "
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
        logger.info(f"Final equity={app.simulate_adapter.get_equity():.4f}")
    finally:
        if app.influx_client is not None:
            app.influx_client.close()


if __name__ == "__main__":
    main()
