import argparse
import os

from src.app.provider import AppProvider
from src.backtest.app import BacktestApp
from src.backtest.data import data_path, download_history, load_candles


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/backtest/")
    return parser.parse_args()


def main():
    args = parse_args()
    provider = AppProvider(mode="backtest", setup_name=args.setup)
    app = BacktestApp(provider)
    logger = app.logger

    path = data_path(app.instrument, app.step, app.backtest_start, app.backtest_end)

    if not os.path.exists(path):
        download_history(
            exchange=provider.okx_exchange,
            start=app.backtest_start,
            end=app.backtest_end,
            step=app.step,
            path=path,
            logger=logger,
        )
    else:
        logger.info(f"Using cached history file {path}")

    try:
        warmup_count = 0
        total = 0
        for trade in load_candles(path):
            total += 1
            try:
                if trade.close is not None:
                    app.exchange.set_price(float(trade.close))
                if warmup_count < app.warmup_periods:
                    app.strategy.warmup(trade)
                    warmup_count += 1
                else:
                    app.strategy.ack(trade)
            except Exception:
                logger.error(
                    f"Strategy ack failed timestamp={trade.timestamp} close={trade.close}"
                )
                raise
        logger.info(
            f"Backtest completed total_candles={total} "
            f"warmup={warmup_count} traded={total - warmup_count}"
        )
        if total == 0:
            logger.warning("No candles returned; strategy ack did not run")
        logger.info(f"Backtest session_id {app.session_id}")
        logger.info(f"Final equity={app.exchange.get_equity():.4f}")
    finally:
        if provider.clickhouse_client is not None:
            provider.clickhouse_client.close()


if __name__ == "__main__":
    main()
