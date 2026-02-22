from src.app.logger import AppLogger
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade
from src.clickhouse.recorder import Recorder
from src.strategy.adapter import StrategyAdapter


class CrossMAStrategy(StrategyAdapter):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, short_length: int,
                  long_length: int):
        super().bootstrap(exchange, short_length, long_length)

    def compute_signal(self, short_ema: float, long_ema: float) -> str | None:
        if short_ema > long_ema:
            return "LONG"
        elif short_ema < long_ema:
            return "SHORT"
        return None

    def ack(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        result = self._signal(trade.price, trade.size)

        equity = self.exchange.get_equity()
        self._logger.info(
            f"CrossMAStrategy bucket "
            f"timestamp={trade.timestamp} "
            f"price={trade.price} "
            f"short_ema={result.short_ema} long_ema={result.long_ema} "
            f"equity={equity:.4f}"
        )

        self._execute(trade, result)
        self._emit_trade_measurement(trade, result)
