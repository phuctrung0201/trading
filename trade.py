"""Live futures trading via OKX WebSocket candle stream."""

from client.influxdb import InfluxClient, InfluxConfig
from client import okx
from dataloader import ohlc
from dataloader.ohlc import Candle as StrategyCandle
from executor.okx import OkxExcutor
from logger import log
import setup


def _build_influx_client() -> InfluxClient | None:
    if not getattr(setup, "influx_enabled", False):
        return None
    config = InfluxConfig(
        url=getattr(setup, "influx_url"),
        token=getattr(setup, "influx_token"),
    )
    return InfluxClient.from_config(config=config)


def preload(client: okx.Client):
    """Fetch recent candles so the strategy has enough history."""
    path = client.prices(
        instrument=setup.instrument,
        bar=setup.step,
        duration_from_now=setup.preload_duration,
    )
    return ohlc.csv(path)


def run(client: okx.Client, executor: OkxExcutor) -> None:
    """Subscribe to live candles and execute the strategy."""
    channel = client.subscribe(instrument=setup.instrument, bar=setup.step)
    try:
        for candle in channel:
            if not candle.confirm:
                log.debug(str(candle))
                continue

            log.info(str(candle))
            executor.ack(
                StrategyCandle(
                    timestamp=candle.timestamp,
                    open=float(candle.open),
                    high=float(candle.high),
                    low=float(candle.low),
                    close=float(candle.close),
                    volume=float(candle.volume),
                )
            )
    finally:
        channel.close()


def main():
    client = setup.okx_client()
    prices = preload(client)
    influx_client = _build_influx_client()
    if influx_client is not None:
        client.enable_request_monitoring(influx_client)

    executor = OkxExcutor(
        instrument=setup.instrument,
        leverage=setup.leverage,
        strategy=setup.strategy,
        okx=client,
        ohlc=prices,
        influx_client=influx_client,
    )

    try:
        run(client, executor)
    except KeyboardInterrupt:
        log.info("Stopping live trading...")
    finally:
        if influx_client is not None:
            influx_client.close()


if __name__ == "__main__":
    main()
