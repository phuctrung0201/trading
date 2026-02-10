from source import binance
from dataloader import order_book, ohlc
from execution import backtester
from strategy import doublema, stoploss

def main():
    pair = "ETH/USDT"
    start = "2025-01-01T00:00:00Z"
    end = "2025-12-01T00:00:00Z"
    step = "1d"
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

    # path = binance.order_book(
    #     pair=pair,
    #     start=start,
    #     end=end,
    #     step=step,
    #     depth="100",
    #     format=format,
    # )
    #
    # book = order_book.csv(path)
    #
    # print(book[:10])


    # The strategy based on a fast MA and a slow MA
    # If the fast MA cross down the slow MA, close long position, open short position
    # If the fast MA cross up the slow MA, open long position, close short position
    double_ma = doublema.Strategy(
        fast=5,
        slow=50,
    )

    sl = stoploss.Strategy(
        percent=10
    )
    
    # Back testing for the strategy
    # Retrun
    # - Max drawdown
    # - Max drawdown duration
    # - Sharp ratio
    # - Profit
    result = backtester.evaluate(
        ohlc = prices,
        # Prior to the first strategy
        strategies = [
            sl,
            double_ma
        ],
        capital = 500,
        print_interval = "7d"
    )

    result.print()

if __name__ == "__main__":
    main()
