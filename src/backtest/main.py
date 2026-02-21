import argparse

from src.app.provider import AppProvider
from src.backtest.app import BacktestApp


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/backtest/")
    return parser.parse_args()


def main():
    args = parse_args()
    provider = AppProvider(mode="backtest", setup_name=args.setup)
    app = BacktestApp(provider)
    logger = app.logger

    try:
        total = 0
        for trade in provider.okx_exchange.stream_history(depth_sec=app.depth_sec):
            total += 1
            try:
                app.exchange.set_price(trade.price)
                app.strategy.ack(trade)
            except Exception:
                logger.error(
                    f"Strategy ack failed timestamp={trade.timestamp} price={trade.price}"
                )
                raise
        logger.info(f"Backtest completed total_trades={total}")
        if total == 0:
            logger.warning("No trades returned; strategy ack did not run")
        logger.info(f"Backtest session_id {app.session_id}")
        logger.info(f"Final equity={app.exchange.get_equity():.4f}")
    finally:
        if provider.clickhouse_client is not None:
            provider.clickhouse_client.close()


if __name__ == "__main__":
    main()
