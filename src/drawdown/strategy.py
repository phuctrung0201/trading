from src.app.logger import AppLogger
from src.crossma.strategy import CrossMAStrategy
from src.drawdown.management import DrawdownManagement
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position
from src.clickhouse.recorder import Recorder
from src.strategy.adapter import SignalResult


class DrawdownCrossMAStrategy(CrossMAStrategy):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)
        self._dd = DrawdownManagement()

    def bootstrap(self, exchange: ExchangeAdapter, short_length: int,
                  long_length: int, bucket_interval: str = "5m",
                  window: int = 500,
                  threshold_scale_map: dict | None = None):
        super().bootstrap(exchange, short_length, long_length, bucket_interval)
        self._dd.bootstrap_drawdown(window, threshold_scale_map)

    def ack(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        bucket = self._accumulate(trade)
        if bucket is None:
            return

        self._tick_count += 1
        self._short_ema.update(bucket.price)
        self._long_ema.update(bucket.price)

        equity = self.exchange.get_equity()
        drawdown = self._dd.calculate_drawdown(equity)
        scale = self._dd.scale_position(drawdown)

        short_ema = self._short_ema.value
        long_ema = self._long_ema.value
        result = SignalResult(short_ema=short_ema, long_ema=long_ema)

        self._logger.info(
            f"DrawdownCrossMAStrategy bucket "
            f"timestamp={bucket.timestamp} "
            f"price={bucket.price} volume={bucket.volume:.4f} "
            f"short_ema={short_ema} long_ema={long_ema} "
            f"equity={equity:.4f} drawdown={drawdown:.4f} scale={scale:.4f}"
        )

        if self._tick_count < self.long:
            self._emit_trade_measurement(trade, result, drawdown=drawdown)
            return

        if not self._warmed_up:
            self._warmed_up = True
            self._logger.info(f"Warmup complete buckets={self._tick_count}")

        if short_ema is None or long_ema is None:
            self._emit_trade_measurement(trade, result, drawdown=drawdown)
            return

        signal = self.compute_signal(short_ema, long_ema)
        result = SignalResult(signal=signal, short_ema=short_ema, long_ema=long_ema)

        if result.signal is not None:
            side = "buy" if result.signal == "LONG" else "sell"

            if self._current_position is not None and self._current_position.side != side:
                close_pnl = self.exchange.unrealized_pnl()
                self._logger.info(
                    f"DrawdownCrossMAStrategy closing side={self._current_position.side} "
                    f"size={self._current_position.size:.4f}"
                )
                self.exchange.close(self._current_position)
                self._emit_event(
                    trade, "close", signal_result=result,
                    fill_price=self._last_close_price,
                    pnl=close_pnl,
                    signal=result.signal,
                    reason=f"signal flip to {result.signal}",
                    drawdown=drawdown,
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
                        f"DrawdownCrossMAStrategy open signal={result.signal} side={side} "
                        f"drawdown={float(drawdown):.4f} scale={float(scale):.4f} "
                        f"equity={equity:.4f} "
                        f"size={self._current_position.size:.4f} "
                        f"fill_price={self._current_position.price}"
                    )
                    self._emit_event(
                        trade, "open", signal_result=result,
                        fill_price=self._current_position.price,
                        signal=result.signal,
                        reason=f"EMA crossover {result.signal} scale={scale:.4f}",
                        drawdown=drawdown,
                        fee=self._last_fee(),
                    )
                else:
                    self._logger.error(
                        f"DrawdownCrossMAStrategy open failed signal={result.signal} side={side} "
                        f"message={open_result.message}"
                    )
                    self._emit_event(
                        trade, "error", signal_result=result,
                        reason=f"open failed: {open_result.message}",
                        drawdown=drawdown,
                    )

        if self._current_position is not None and scale != self._exposure_ratio:
            old_scale = self._exposure_ratio
            self._logger.info(
                f"DrawdownCrossMAStrategy scale changed "
                f"old_scale={old_scale:.4f} new_scale={scale:.4f} "
                f"resizing position"
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
                    fee=self._last_fee(),
                )
            else:
                self._logger.error(
                    f"DrawdownCrossMAStrategy resize failed side={side} "
                    f"message={open_result.message}"
                )
                self._emit_event(
                    trade, "error", signal_result=result,
                    reason=f"resize failed: {open_result.message}",
                    drawdown=drawdown,
                )

        self._emit_trade_measurement(trade, result, drawdown=drawdown)
