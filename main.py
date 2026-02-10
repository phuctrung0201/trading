from source import binance
from dataloader import order_book, ohlc
from execution import backtester
from strategy import macross, buydip, diprecover, drawdownscale, equityguard, stoploss

def main():
    pair = "ETH/USDT"
    start = "2025-01-01T00:00:00Z"
    end = "2025-04-01T00:00:00Z"
    step = "1m"
    format = "csv"

    path = binance.price(
        pair=pair,
        start=start,
        end=end,
        step=step,
        format=format,
    )
    prices = ohlc.csv(path)

    print(prices[:10])

    # -- Stop loss overlay --
    sl = stoploss.Strategy(
        percent=10
    )

    # -- MA Cross strategy --
    m = macross.Strategy(
        short=5,
        long=10,
        source="close",
    )

    print("\n=== MA Cross Strategy ===")
    result_ma = backtester.evaluate(
        ohlc = prices,
        strategies = [sl, m],
        capital = 500,
    )
    result_ma.print()
    result_ma.plot("result_macross.png")

    # -- Buy-the-Dip Mean Reversion strategy --
    # Go long when price dips 1.5% from the rolling 60-bar high
    # Go short when price rallies 1.5% from the rolling 60-bar low
    dip = buydip.Strategy(
        lookback=60,
        dip_pct=1.5,
        rally_pct=1.5,
    )

    print("\n=== Buy-the-Dip Mean Reversion Strategy ===")
    result_dip = backtester.evaluate(
        ohlc = prices,
        strategies = [sl, dip],
        capital = 500,
    )
    result_dip.print()
    result_dip.plot("result_buydip.png")

    # -- Dip Recovery strategy --
    # Wait for a 2% dip, then enter long only after a 0.5% bounce confirms recovery.
    # Mirror for shorts: wait for a 2% rally, then short on 0.5% pullback.
    dr = diprecover.Strategy(
        lookback=60,
        dip_pct=2.0,
        recovery_pct=0.5,
        rally_pct=2.0,
        pullback_pct=0.5,
    )

    print("\n=== Dip Recovery Strategy ===")
    result_dr = backtester.evaluate(
        ohlc = prices,
        strategies = [sl, dr],
        capital = 500,
    )
    result_dr.print()
    result_dr.plot("result_diprecover.png")

    # -- Drawdown Scaling strategy --
    # Scales position size with dip depth (DCA/grid approach).
    # Starts entering at 0.5% dip, reaches max 2x size at 3% dip.
    ds = drawdownscale.Strategy(
        lookback=60,
        dip_entry_pct=0.5,
        rally_entry_pct=0.5,
        full_scale_pct=3.0,
        max_scale=2.0,
    )

    print("\n=== Drawdown Scaling Strategy ===")
    result_ds = backtester.evaluate(
        ohlc = prices,
        strategies = [sl, ds],
        capital = 500,
    )
    result_ds.print()
    result_ds.plot("result_drawdownscale.png")

    # -- Equity Guard overlay on Dip Recovery --
    # Uses Dip Recovery as the base strategy, but pauses trading
    # when equity draws down 5% from its peak. Resumes at -2%.
    eg = equityguard.Strategy(
        max_dd_pct=5.0,
        resume_pct=2.0,
    )

    print("\n=== Dip Recovery + Equity Guard ===")
    result_eg = backtester.evaluate(
        ohlc = prices,
        # Equity guard sits before stoploss in the overlay chain
        # Pipeline: dr (base) → eg (equity guard) → sl (stoploss)
        strategies = [sl, eg, dr],
        capital = 500,
    )
    result_eg.print()
    result_eg.plot("result_equityguard.png")

if __name__ == "__main__":
    main()
