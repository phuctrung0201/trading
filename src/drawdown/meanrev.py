from src.app.logger import AppLogger
from src.clickhouse.recorder import Recorder
from src.drawdown.management import DrawdownManagement
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position
from src.meanrev.bucket import MeanRevStrategy
from src.strategy.adapter import SignalResult


class DrawdownMeanRevStrategy(MeanRevStrategy):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)
        self._dd = DrawdownManagement()

    def bootstrap(self, exchange: ExchangeAdapter, bucket_interval: str = "5m",
                  window: int = 500, threshold_scale_map: dict | None = None,
                  lookback: int = 100, entry_threshold: float = 2.0,
                  exit_threshold: float = 0.5):
        super().bootstrap(exchange, bucket_interval, lookback,
                          entry_threshold, exit_threshold)
        self._dd.bootstrap_drawdown(window, threshold_scale_map)

    def ack(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        bucket = self._accumulate(trade)
        if bucket is None:
            return

        equity = self.exchange.get_equity()
        drawdown = self._dd.calculate_drawdown(equity)
        scale = self._dd.scale_position(drawdown)

        signal = self.meanrev.push(bucket.price)
        z = self.meanrev.last_z
        result = SignalResult(signal=signal)

        self._logger.info(
            f"DrawdownMeanRevStrategy bucket "
            f"timestamp={bucket.timestamp} "
            f"price={bucket.price} volume={bucket.volume:.4f} "
            f"z={z} "
            f"equity={equity:.4f} drawdown={drawdown:.4f} scale={scale:.4f}"
        )

        if not self.meanrev.is_ready():
            self._emit_trade_measurement(trade, result, drawdown=drawdown, zscore=z)
            return

        if signal == "EXIT" and self._current_position is not None:
            close_pnl = self.exchange.unrealized_pnl()
            self._logger.info(
                f"DrawdownMeanRevStrategy closing (revert) "
                f"side={self._current_position.side} "
                f"size={self._current_position.size:.4f}"
            )
            self.exchange.close(self._current_position)
            self._emit_event(
                trade, "close", signal_result=result,
                fill_price=self._last_close_price,
                pnl=close_pnl,
                signal="EXIT",
                reason=f"z={z:.4f} reverted",
                drawdown=drawdown,
                zscore=z,
                fee=self._last_fee(),
            )
            self._current_position = None
            self._exposure_ratio = 1.0

        elif signal in ("LONG", "SHORT"):
            side = "buy" if signal == "LONG" else "sell"

            if self._current_position is not None and self._current_position.side != side:
                close_pnl = self.exchange.unrealized_pnl()
                self._logger.info(
                    f"DrawdownMeanRevStrategy closing (flip) "
                    f"side={self._current_position.side} "
                    f"size={self._current_position.size:.4f}"
                )
                self.exchange.close(self._current_position)
                self._emit_event(
                    trade, "close", signal_result=result,
                    fill_price=self._last_close_price,
                    pnl=close_pnl,
                    signal=signal,
                    reason=f"signal flip to {signal} z={z:.4f}",
                    drawdown=drawdown,
                    zscore=z,
                    fee=self._last_fee(),
                )
                self._current_position = None
                self._exposure_ratio = 1.0

            if self._current_position is None:
                self._exposure_ratio = scale
                equity = self.exchange.get_equity()
                size = equity * float(scale)
                position = Position(side=side, size=size)
                open_result = self.exchange.open(position)
                if open_result.success:
                    self._current_position = open_result.position or position
                    self._logger.info(
                        f"DrawdownMeanRevStrategy open signal={signal} side={side} "
                        f"z={z:.4f} drawdown={drawdown:.4f} scale={scale:.4f} "
                        f"equity={equity:.4f} "
                        f"size={self._current_position.size:.4f} "
                        f"fill_price={self._current_position.price}"
                    )
                    self._emit_event(
                        trade, "open", signal_result=result,
                        fill_price=self._current_position.price,
                        signal=signal,
                        reason=f"z={z:.4f} scale={scale:.4f}",
                        drawdown=drawdown,
                        zscore=z,
                        fee=self._last_fee(),
                    )
                else:
                    self._logger.error(
                        f"DrawdownMeanRevStrategy open failed signal={signal} "
                        f"side={side} message={open_result.message}"
                    )
                    self._emit_event(
                        trade, "error", signal_result=result,
                        reason=f"open failed: {open_result.message}",
                        drawdown=drawdown,
                        zscore=z,
                    )

        if self._current_position is not None and scale != self._exposure_ratio:
            old_scale = self._exposure_ratio
            self._logger.info(
                f"DrawdownMeanRevStrategy scale changed "
                f"old_scale={old_scale:.4f} new_scale={scale:.4f}"
            )
            side = self._current_position.side
            close_pnl = self.exchange.unrealized_pnl()
            self.exchange.close(self._current_position)
            self._emit_event(
                trade, "close", signal_result=result,
                fill_price=self._last_close_price,
                pnl=close_pnl,
                reason=f"resize close old_scale={old_scale:.4f}",
                drawdown=drawdown,
                zscore=z,
                fee=self._last_fee(),
            )
            self._current_position = None

            self._exposure_ratio = scale
            equity = self.exchange.get_equity()
            size = equity * float(scale)
            position = Position(side=side, size=size)
            open_result = self.exchange.open(position)
            if open_result.success:
                self._current_position = open_result.position or position
                self._emit_event(
                    trade, "resize", signal_result=result,
                    fill_price=self._current_position.price,
                    reason=f"drawdown scale {old_scale:.4f} -> {scale:.4f}",
                    drawdown=drawdown,
                    zscore=z,
                    fee=self._last_fee(),
                )
            else:
                self._logger.error(
                    f"DrawdownMeanRevStrategy resize failed side={side} "
                    f"message={open_result.message}"
                )
                self._emit_event(
                    trade, "error", signal_result=result,
                    reason=f"resize failed: {open_result.message}",
                    drawdown=drawdown,
                    zscore=z,
                )

        self._emit_trade_measurement(trade, result, drawdown=drawdown, zscore=z)
