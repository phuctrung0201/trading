from src.app.logger import AppLogger
from src.clickhouse.recorder import Recorder
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade
from src.strategy.adapter import StrategyAdapter, _parse_interval


class BucketStrategy(StrategyAdapter):
    """Accumulates trades into fixed time buckets.

    Subclasses call ``_accumulate(trade)`` on every tick.  When the bucket
    rolls over, the method returns the price with the largest total volume
    in the previous bucket.  Returns ``None`` while still inside the same
    bucket.
    """

    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, bucket_interval: str = "5m"):
        super().bootstrap(exchange)
        self._bucket_ms = _parse_interval(bucket_interval) * 1000
        self._current_bucket: int | None = None
        self._bucket_volumes: dict[float, float] = {}

    def _accumulate(self, trade: MarketTrade) -> float | None:
        ts_ms = int(trade.timestamp)
        bucket_idx = ts_ms // self._bucket_ms

        if self._current_bucket is None:
            self._current_bucket = bucket_idx
            self._bucket_volumes[trade.price] = trade.size
            return None

        if bucket_idx == self._current_bucket:
            self._bucket_volumes[trade.price] = (
                self._bucket_volumes.get(trade.price, 0.0) + trade.size
            )
            return None

        bucket_price = max(
            self._bucket_volumes.items(), key=lambda kv: kv[1],
        )[0]

        self._current_bucket = bucket_idx
        self._bucket_volumes.clear()
        self._bucket_volumes[trade.price] = trade.size

        return bucket_price
