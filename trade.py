import argparse
import logging
import os
import signal
import subprocess
import sys

from src.app.config import list_setups
from src.app.trade import TradeApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument("--setup", type=str, default=None, help="Setup name from setup/ folder")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.setup:
        run_trade(args.setup)
    else:
        setups = list_setups()
        if not setups:
            run_trade(None)
            return
        run_all_setups(setups)


def run_all_setups(setups: list[str]):
    logger = logging.getLogger("trading.app")
    children: list[subprocess.Popen] = []

    for name in setups:
        logger.info(f"Spawning trade process for setup={name}")
        proc = subprocess.Popen(
            [sys.executable, "-u", "trade.py", "--setup", name],
            cwd=os.getcwd(),
        )
        children.append(proc)

    def forward_signal(signum, _frame):
        for proc in children:
            if proc.poll() is None:
                proc.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    for proc in children:
        proc.wait()


def run_trade(setup_name: str | None):
    app = TradeApp(setup_name=setup_name)
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
