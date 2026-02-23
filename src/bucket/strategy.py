from dataclasses import dataclass

from src.app.logger import AppLogger
from src.clickhouse.recorder import Recorder
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade
from src.strategy.adapter import StrategyAdapter, _parse_interval


@dataclass
class Bucket:
    price: float
    timestamp: int
    volume: float


class BucketStrategy(StrategyAdapter):
    """Accumulates trades into fixed time buckets.

    Subclasses call ``_accumulate(trade)`` on every tick.  When the bucket
    rolls over, the method returns a ``Bucket`` with the VWAP, the bucket
    open timestamp, and the total volume.  Returns ``None`` while still
    inside the same bucket.
    """

    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, bucket_interval: str = "5m"):
        super().bootstrap(exchange)
        self._bucket_ms = _parse_interval(bucket_interval) * 1000
        self._current_bucket: int | None = None
        self._bucket_pv_sum: float = 0.0
        self._bucket_total_volume: float = 0.0

    def _accumulate(self, trade: MarketTrade) -> Bucket | None:
        ts_ms = int(trade.timestamp)
        bucket_idx = ts_ms // self._bucket_ms

        if self._current_bucket is None:
            self._current_bucket = bucket_idx
            self._bucket_pv_sum = trade.price * trade.size
            self._bucket_total_volume = trade.size
            return None

        if bucket_idx == self._current_bucket:
            self._bucket_pv_sum += trade.price * trade.size
            self._bucket_total_volume += trade.size
            return None

        vwap = self._bucket_pv_sum / self._bucket_total_volume
        bucket_ts = self._current_bucket * self._bucket_ms
        bucket_volume = self._bucket_total_volume

        self._current_bucket = bucket_idx
        self._bucket_pv_sum = trade.price * trade.size
        self._bucket_total_volume = trade.size

        return Bucket(price=vwap, timestamp=bucket_ts, volume=bucket_volume)
