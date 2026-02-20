from src.app.config import AppConfig
from src.app.logger import AppLogger
from src.client.ohclv import OHCLV
from src.measurement.trade import TradeMeasurement
from src.app.adapter import ExchangeAdapter, Position
from src.strategy.crossma import CrossMAStrategy


class DrawdownStrategy(CrossMAStrategy):
    def init_drawdown(self, window):
        self.drawdown_window = int(window)
        self._equity_window: list[float] = []

    def init_threshold(self, threshold: dict[float, float]):
        self.position_sizing_map = {float(k): float(v) for k, v in threshold.items()}
        self.drawdown_thresholds = sorted(self.position_sizing_map.keys())

    def __init__(
        self,
        exchange_adapter: ExchangeAdapter,
        measurement_adapter,
        app_logger: AppLogger,
        app_config: AppConfig,
    ):
        super().__init__(
            exchange_adapter, measurement_adapter, app_logger=app_logger,
            app_config=app_config,
        )
        drawdown_config = app_config.values.drawdown
        self.init_drawdown(window=drawdown_config.window)
        self.init_threshold(threshold=drawdown_config.threshold_scale_map)

    def calculate_drawdown(self):
        equity = self.exchange_adapter.get_equity()
        self._equity_window.append(equity)
        if len(self._equity_window) > self.drawdown_window:
            self._equity_window = self._equity_window[-self.drawdown_window :]
        peak = max(self._equity_window)
        if peak <= 0:
            return 0.0
        return (equity - peak) / peak

    def scale_position(self, drawdown):
        dd = abs(drawdown)
        scale = float(self.position_sizing_map.get(0.0, 1.0))
        for threshold in self.drawdown_thresholds:
            if dd >= threshold:
                scale = float(self.position_sizing_map[threshold])
        return scale

    def ack(self, candle: OHCLV):
        self.reconcile()
        self._mark_to_market()
        drawdown = self.calculate_drawdown()
        scale = self.scale_position(drawdown)
        result = self._signal(candle)

        equity = self.exchange_adapter.get_equity()
        self._logger.info(
            f"DrawdownStrategy candle "
            f"timestamp={getattr(candle, 'timestamp', None)} "
            f"open={getattr(candle, 'open', None)} "
            f"high={getattr(candle, 'high', None)} "
            f"low={getattr(candle, 'low', None)} "
            f"close={getattr(candle, 'close', None)} "
            f"volume={getattr(candle, 'volume', None)} "
            f"short_ema={result.short_ema} long_ema={result.long_ema} "
            f"equity={equity:.4f} drawdown={drawdown:.4f} scale={scale:.4f}"
        )

        if result.signal is not None:
            side = "buy" if result.signal == "LONG" else "sell"

            if self._current_position is not None and self._current_position.side != side:
                close_pnl = self.exchange_adapter._calculate_unrealized_pnl() if hasattr(self.exchange_adapter, '_calculate_unrealized_pnl') else 0.0
                self._logger.info(
                    f"DrawdownStrategy closing side={self._current_position.side} "
                    f"size={self._current_position.size:.4f}"
                )
                self.exchange_adapter.close(self._current_position)
                self._emit_event(
                    candle, "close", signal_result=result,
                    fill_price=self._last_close_price,
                    pnl=close_pnl,
                    signal=result.signal,
                    reason=f"signal flip to {result.signal}",
                )
                self._current_position = None
                self._exposure_ratio = 1.0

            if self._current_position is None:
                self._exposure_ratio = scale
                equity = self.exchange_adapter.get_equity()
                size = equity * float(scale)
                position = Position(side=side, size=size)
                open_result = self.exchange_adapter.open(position)
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
                        candle, "open", signal_result=result,
                        fill_price=self._current_position.price,
                        signal=result.signal,
                        reason=f"EMA crossover {result.signal} scale={scale:.4f}",
                    )
                else:
                    self._logger.error(
                        f"DrawdownStrategy open failed signal={result.signal} side={side} "
                        f"message={open_result.message}"
                    )
                    self._emit_event(
                        candle, "error", signal_result=result,
                        reason=f"open failed: {open_result.message}",
                    )

        if self._current_position is not None and scale != self._exposure_ratio:
            self._logger.info(
                f"DrawdownStrategy scale changed "
                f"old_scale={self._exposure_ratio:.4f} new_scale={scale:.4f} "
                f"resizing position"
            )
            side = self._current_position.side
            self.exchange_adapter.close(self._current_position)
            self._current_position = None

            self._exposure_ratio = scale
            equity = self.exchange_adapter.get_equity()
            size = equity * float(scale)
            position = Position(side=side, size=size)
            open_result = self.exchange_adapter.open(position)
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
                    candle, "resize", signal_result=result,
                    fill_price=self._current_position.price,
                    reason=f"drawdown scale {self._exposure_ratio:.4f} -> {scale:.4f}",
                )
            else:
                self._logger.error(
                    f"DrawdownStrategy resize failed side={side} "
                    f"message={open_result.message}"
                )
                self._emit_event(
                    candle, "error", signal_result=result,
                    reason=f"resize failed: {open_result.message}",
                )

        close_price = float(getattr(candle, "close", 0) or 0)
        self._last_close_price = close_price
        position_size = (
            float(self._current_position.size) if self._current_position is not None else 0.0
        )
        position_side = self._current_position.side if self._current_position is not None else "flat"
        if position_side == "sell":
            position_size = -position_size
        trade_measurement = TradeMeasurement(
            timestamp=self._measurement_timestamp(candle),
            equity=self.exchange_adapter.get_equity(),
            position_size=position_size,
            position_side=position_side,
            drawdown=float(drawdown),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            short_ema=result.short_ema,
            long_ema=result.long_ema,
            close_price=close_price,
            exposure_ratio=self._exposure_ratio,
        )
        self.measurement_adapter.record(trade_measurement)
