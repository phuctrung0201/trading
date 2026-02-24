from src.app.logger import AppLogger
from src.bucket.strategy import BucketStrategy
from src.clickhouse.recorder import Recorder
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade
from src.meanrev.strategy import MeanRevIndicator
from src.strategy.adapter import SignalResult


class MeanRevStrategy(BucketStrategy):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, bucket_interval: str = "5m",
                  lookback: int = 100, entry_threshold: float = 2.0,
                  exit_threshold: float = 0.5):
        super().bootstrap(exchange, bucket_interval)
        self._meanrev = MeanRevIndicator(lookback, entry_threshold, exit_threshold)

    def _signal(self, trade: MarketTrade) -> SignalResult:
        bucket = self._accumulate(trade)
        if bucket is None:
            return SignalResult()

        signal = self._meanrev.push(bucket.price)
        z = self._meanrev.last_z
        return SignalResult(signal=signal)

    @property
    def meanrev(self) -> MeanRevIndicator:
        return self._meanrev

    def ack_trade(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        result = self._signal(trade)
        z = self._meanrev.last_z

        if result.signal is None and z is None:
            return

        equity = self.exchange.get_equity()
        self._logger.info(
            f"MeanRevStrategy bucket "
            f"timestamp={trade.timestamp} "
            f"price={trade.price} z={z} "
            f"equity={equity:.4f}"
        )

        if result.signal == "EXIT" and self._current_position is not None:
            close_pnl = self.exchange.unrealized_pnl()
            self.exchange.close(self._current_position)
            self._emit_event(
                trade, "close", signal_result=result,
                fill_price=self._last_close_price,
                pnl=close_pnl,
                signal="EXIT",
                reason=f"z={z:.4f} reverted",
                zscore=z,
                fee=self._last_fee(),
            )
            self._current_position = None
        else:
            self._execute(trade, result)

        self._emit_trade_measurement(trade, result, zscore=z)
