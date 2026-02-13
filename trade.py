"""Live futures trading via OKX WebSocket candle stream."""

from source import okx
from dataloader import ohlc
from executor.future import Future
from logger import log
import setup


def connect() -> okx.Client:
    """Authenticate and return an OKX client."""
    return okx.Client(
        api_key=setup.okx_api_key,
        secret_key=setup.okx_secret_key,
        passphrase=setup.okx_passphrase,
        demo=setup.okx_demo,
    )


def preload(client: okx.Client):
    """Fetch recent candles so the strategy has enough history."""
    path = client.prices(
        instrument=setup.instrument,
        bar=setup.step,
        duration_from_now=setup.preload_duration,
    )
    return ohlc.csv(path)


def run(client: okx.Client, executor: Future) -> None:
    """Subscribe to live candles and execute the strategy."""
    channel = client.subscribe(instrument=setup.instrument, bar=setup.step)

    for candle in channel:
        if candle.confirm:
            log.info(candle)
        else:
            log.debug(candle)

        executor.ack(candle)


def main():
    client = connect()
    prices = preload(client)
    capital = client.asset("USDT")

    executor = Future(
        cap=capital,
        instrument=setup.instrument,
        leverage=setup.leverage,
        strategy=setup.strategy,
        okx=client,
        ohlc=prices,
    )

    run(client, executor)


if __name__ == "__main__":
    main()
