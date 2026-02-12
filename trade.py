from source import okx
from dataloader import ohlc
from execution.future import Future
import setup

def main():
    client = okx.Client(
        api_key=setup.okx_api_key,
        secret_key=setup.okx_secret_key,
        passphrase=setup.okx_passphrase,
        demo=setup.okx_demo,
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
        strategy=setup.strategy,
        okx=client,
        ohlc=prices,
    )

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
