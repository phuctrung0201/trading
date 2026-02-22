from collections import deque

from src.app.logger import AppLogger
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position
from src.clickhouse.recorder import Recorder
from src.strategy.adapter import StrategyAdapter


class DrawdownStrategy(StrategyAdapter):
    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)

    def bootstrap(self, exchange: ExchangeAdapter, short_length: int,
                  long_length: int, window: int = 500,
                  threshold_scale_map: dict | None = None):
        super().bootstrap(exchange, short_length, long_length)
        self.drawdown_window = int(window)
        self._equity_window: deque[float] = deque(maxlen=self.drawdown_window)
        raw_map = threshold_scale_map or {0.0: 1.0}
        self.position_sizing_map = {float(k): float(v) for k, v in raw_map.items()}
        self.drawdown_thresholds = sorted(self.position_sizing_map.keys())

    def compute_signal(self, short_ema: float, long_ema: float) -> str | None:
        if short_ema > long_ema:
            return "LONG"
        elif short_ema < long_ema:
            return "SHORT"
        return None

    def calculate_drawdown(self, equity: float) -> float:
        self._equity_window.append(equity)
        peak = max(self._equity_window)
        if peak <= 0:
            return 0.0
        return (equity - peak) / peak

    def scale_position(self, drawdown: float) -> float:
        dd = abs(drawdown)
        scale = float(self.position_sizing_map.get(0.0, 1.0))
        for threshold in self.drawdown_thresholds:
            if dd >= threshold:
                scale = float(self.position_sizing_map[threshold])
        return scale

    def ack(self, trade: MarketTrade):
        self.exchange.set_price(trade.price)
        self._mark_to_market()
        self.reconcile()

        equity = self.exchange.get_equity()
        drawdown = self.calculate_drawdown(equity)
        scale = self.scale_position(drawdown)
        result = self._signal(trade.price)
        self._logger.info(
            f"DrawdownStrategy bucket "
            f"timestamp={trade.timestamp} "
            f"price={trade.price} "
            f"short_ema={result.short_ema} long_ema={result.long_ema} "
            f"equity={equity:.4f} drawdown={drawdown:.4f} scale={scale:.4f}"
        )

        if result.signal is not None:
            side = "buy" if result.signal == "LONG" else "sell"

            if self._current_position is not None and self._current_position.side != side:
                close_pnl = self.exchange.unrealized_pnl()
                self._logger.info(
                    f"DrawdownStrategy closing side={self._current_position.side} "
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
                        f"DrawdownStrategy open signal={result.signal} side={side} "
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
                    )
                else:
                    self._logger.error(
                        f"DrawdownStrategy open failed signal={result.signal} side={side} "
                        f"message={open_result.message}"
                    )
                    self._emit_event(
                        trade, "error", signal_result=result,
                        reason=f"open failed: {open_result.message}",
                        drawdown=drawdown,
                    )

        if self._current_position is not None and scale != self._exposure_ratio:
            self._logger.info(
                f"DrawdownStrategy scale changed "
                f"old_scale={self._exposure_ratio:.4f} new_scale={scale:.4f} "
                f"resizing position"
            )
            side = self._current_position.side
            self.exchange.close(self._current_position)
            self._current_position = None

            self._exposure_ratio = scale
            equity = self.exchange.get_equity()
            size = equity * float(scale)
            position = Position(side=side, size=size)
            open_result = self.exchange.open(position)
            if open_result.success:
                self._current_position = open_result.position or position
                self._logger.info(
                    f"DrawdownStrategy resized side={side} "
                    f"scale={float(scale):.4f} "
                    f"equity={equity:.4f} "
                    f"size={self._current_position.size:.4f} "
                    f"fill_price={self._current_position.price}"
                )
                self._emit_event(
                    trade, "resize", signal_result=result,
                    fill_price=self._current_position.price,
                    reason=f"drawdown scale {self._exposure_ratio:.4f} -> {scale:.4f}",
                    drawdown=drawdown,
                )
            else:
                self._logger.error(
                    f"DrawdownStrategy resize failed side={side} "
                    f"message={open_result.message}"
                )
                self._emit_event(
                    trade, "error", signal_result=result,
                    reason=f"resize failed: {open_result.message}",
                    drawdown=drawdown,
                )

        self._emit_trade_measurement(trade, result, drawdown=drawdown)
