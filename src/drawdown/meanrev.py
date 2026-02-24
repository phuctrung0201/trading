from src.app.logger import AppLogger
from src.clickhouse.recorder import Recorder
from src.drawdown.strategy import DrawdownStrategy
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade
from src.meanrev.strategy import MeanRevIndicator
from src.strategy.adapter import SignalResult


class DrawdownMeanRevStrategy(DrawdownStrategy):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, bucket_interval: str = "5m",
                  window: int = 500, threshold_scale_map: dict | None = None,
                  lookback: int = 100, entry_threshold: float = 2.0,
                  exit_threshold: float = 0.5):
        super().bootstrap(exchange, bucket_interval, window, threshold_scale_map)
        self._meanrev = MeanRevIndicator(lookback, entry_threshold, exit_threshold)

    def ack_trade(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        bucket = self._accumulate(trade)
        if bucket is None:
            return

        equity, drawdown, scale = self._drawdown_and_scale()

        signal = self._meanrev.push(bucket.price)
        z = self._meanrev.last_z
        result = SignalResult(signal=signal)

        self._logger.info(
            f"DrawdownMeanRevStrategy bucket "
            f"timestamp={bucket.timestamp} "
            f"price={bucket.price} volume={bucket.volume:.4f} "
            f"z={z} "
            f"equity={equity:.4f} drawdown={drawdown:.4f} scale={scale:.4f}"
        )

        if not self._meanrev.is_ready():
            self._emit_trade_measurement(trade, result, drawdown=drawdown, zscore=z)
            return

        self._execute_drawdown(trade, result, drawdown, scale, zscore=z)
        self._emit_trade_measurement(trade, result, drawdown=drawdown, zscore=z)
