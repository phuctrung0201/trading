from dataloader.ohlc import Candle
from strategy.noaction import NoActionStrategy


class NoActionExecution:
    def ack(self, candle: Candle):
       pass 
