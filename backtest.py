from client import binance
from client.influxdb import InfluxClient, InfluxConfig
from dataloader import ohlc
from dataloader.ohlc import Candle
from executor.backtester import Backtester
import setup


def _build_influx_client() -> InfluxClient | None:
    if not getattr(setup, "influx_enabled", False):
        return None
    config = InfluxConfig(
        url=getattr(setup, "influx_url"),
        token=getattr(setup, "influx_token"),
    )
    return InfluxClient.from_config(config=config)


def main():
    path = binance.price(
        instrument=setup.instrument,
        start=setup.start,
        end=setup.end,
        step=setup.step,
        format="csv",
    )
    candles = ohlc.csv(path)

    backtester = Backtester(
        strategy=setup.strategy,
        influx_client=_build_influx_client(),
    )
    try:
        for row in candles.itertuples(index=True):
            backtester.ack(
                Candle(
                    timestamp=str(row.Index),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    timestamp_ns=int(row.Index.value),
                )
            )
        backtester.summary()
        backtester.result("backtest.png")
    finally:
        backtester.close()

if __name__ == "__main__":
    main()
