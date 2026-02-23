from src.app.logger import AppLogger
from src.bucket.strategy import BucketStrategy
from src.ema.indicator import EMA
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade
from src.clickhouse.recorder import Recorder
from src.strategy.adapter import SignalResult


class CrossMAStrategy(BucketStrategy):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, short_length: int,
                  long_length: int, bucket_interval: str = "5m"):
        super().bootstrap(exchange, bucket_interval)
        self.short = int(short_length)
        self.long = int(long_length)
        self._short_ema = EMA(period=self.short)
        self._long_ema = EMA(period=self.long)
        self._tick_count: int = 0
        self._warmed_up: bool = False

    def compute_signal(self, short_ema: float, long_ema: float) -> str | None:
        if short_ema > long_ema:
            return "LONG"
        elif short_ema < long_ema:
            return "SHORT"
        return None

    def _signal(self, trade: MarketTrade) -> SignalResult:
        bucket = self._accumulate(trade)
        if bucket is None:
            return SignalResult(
                short_ema=self._short_ema.value,
                long_ema=self._long_ema.value,
            )

        self._tick_count += 1
        self._short_ema.update(bucket.price)
        self._long_ema.update(bucket.price)

        if self._tick_count < self.long:
            self._logger.info(f"Warmup {self._tick_count}/{self.long}")
            return SignalResult(
                short_ema=self._short_ema.value,
                long_ema=self._long_ema.value,
            )

        if not self._warmed_up:
            self._warmed_up = True
            self._logger.info(f"Warmup complete buckets={self._tick_count}")

        short_ema = self._short_ema.value
        long_ema = self._long_ema.value
        if short_ema is None or long_ema is None:
            return SignalResult()

        signal = self.compute_signal(short_ema, long_ema)
        return SignalResult(signal=signal, short_ema=short_ema, long_ema=long_ema)

    def ack(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        result = self._signal(trade)

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
