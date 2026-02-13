from source import binance
from dataloader import ohlc
from dataloader.ohlc import Candle
from executor.backtester import Backtester
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

    backtester = Backtester(strategy=setup.strategy)
    for _, row in candles.iterrows():
        backtester.ack(Candle.from_series(row))
    backtester.summary()
    backtester.result("backtest.png")

if __name__ == "__main__":
    main()
