from source import okx
from execution import future
from strategy import stoploss, drawdownscale
from dataloader import ohlc

if __name__ == "__main__":
    client = okx.Client(
        api_key="d009e341-3d49-4f55-b198-548281f1f3b5",
        secret_key="136E730027D0803623471CCCDCD54809",
        passphrase="Nothing0@0!",
        demo=True,
    )

    instrument = "ETH-USDT-SWAP"

    capital = client.asset("USDT")
    preload = client.prices(instrument=instrument, bar="1m", duration_from_now="6h")
    prices = ohlc.csv(preload) 
    context = future.NewContext(
        capital = capital,
        ohlc = prices,
        instrument = instrument,
        leverage = "10",
    )

    ds = drawdownscale.Strategy(
        lookback=60,
        dip_entry_pct=0.5,
        rally_entry_pct=0.5,
        full_scale_pct=3.0,
        max_scale=2.0,
    )

    sl = stoploss.Strategy(
        percent=10
    )

    # Stream real-time candles. Blocks until Ctrl-C.
    channel = client.subscribe(instrument=instrument, bar="1m")
    for candle in channel:
        print(candle)
        context.update(candle)
        if candle.confirm:
            result = future.evaluate(
                context = context,
                okx = client,
                strategies = [
                    sl,
                    ds
                ],
                value_percent = 10
            )
            if result:
                print(result)
