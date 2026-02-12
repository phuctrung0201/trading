from source import okx
from dataloader import ohlc
from execution.future import Future
import setup

def main():
    client = okx.Client(
        api_key="d009e341-3d49-4f55-b198-548281f1f3b5",
        secret_key="136E730027D0803623471CCCDCD54809",
        passphrase="Nothing0@0!",
        demo=True,
    )

    capital = client.asset("USDT")
    preload = client.prices(
        instrument=setup.instrument,
        bar=setup.step,
        duration_from_now=setup.preload_duration
    )
    prices = ohlc.csv(preload)

    executor = Future(
        cap=capital,
        instrument=setup.instrument,
        leverage=setup.leverage,
        okx=client,
        ohlc=prices
    )

    executor.set_entry(setup.entry)

    # Subscribe to live candles (blocks until Ctrl-C)
    channel = client.subscribe(instrument=setup.instrument, bar=setup.step)
    for candle in channel:
        print(candle)
        executor.ack(candle)
        if candle.confirm:
            result = executor.exec()
            if result:
                print(result)

if __name__ == "__main__":
    main()
