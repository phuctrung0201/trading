from source import binance
from dataloader import order_book, ohlc
from execution.backtester import Backtester
import setup

def main():
    path = binance.price(
        instrument=setup.instrument,
        start=setup.start,
        end=setup.end,
        step=setup.step,
        format="csv",
    )
    candles = ohlc.csv(path)

    print(candles[:10])

    backtester = Backtester(
        cap=setup.cap,
        strategy=setup.strategy,
    )

    for _, candle in candles.iterrows():
        backtester.ack(candle)
        backtester.exec()

    backtester.result("backtest.png")

if __name__ == "__main__":
    main()
