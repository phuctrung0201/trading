from src.app.logger import AppLogger
from src.client.ohclv import OHCLV
from src.measurement.trade import TradeMeasurement
from src.strategy.adapter import ExchangeAdapter, Position
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
        periods_per_year: int = 525600,
    ):
        super().__init__(
            exchange_adapter, measurement_adapter, app_logger=app_logger,
            periods_per_year=periods_per_year,
        )
        self.init_drawdown(window=500)
        self.init_threshold(
            threshold={
                0.0: 1.0,
                0.05: 0.5,
                0.10: 0.25,
                0.20: 0.10,
            }
        )

    def calculate_drawdown(self, candle):
        _ = candle
        self._equity_window.append(float(self.equity))
        if len(self._equity_window) > self.drawdown_window:
            self._equity_window = self._equity_window[-self.drawdown_window :]
        if not self._equity_window:
            return 0.0
        peak = max(self._equity_window)
        if peak <= 0:
            return 0.0
        return (peak - float(self.equity)) / peak

    def scale_position(self, drawdown):
        scale = float(self.position_sizing_map.get(0.0, 1.0))
        for threshold in self.drawdown_thresholds:
            if drawdown >= threshold:
                scale = float(self.position_sizing_map[threshold])
        return scale

    def ack(self, candle: OHCLV):
        timestamp = getattr(candle, "timestamp", None)
        self._logger.info(f"DrawdownStrategy ack timestamp={timestamp}")
        self._mark_to_market(candle)
        signal = self._signal(candle)
        if signal is None:
            self._logger.info("DrawdownStrategy no signal")
            return

        drawdown = self.calculate_drawdown(candle)
        scale = self.scale_position(drawdown)
        side = "buy" if signal == "LONG" else "sell"

        if self._current_position is not None and self._current_position.side != side:
            self.exchange_adapter.close(self._current_position)
            self._current_position = None

        if self._current_position is None:
            size = float(self.equity) * float(scale)
            position = Position(side=side, size=size)
            result = self.exchange_adapter.open(position)
            if result.success:
                self._current_position = result.position or position

        position_size = (
            float(self._current_position.size) if self._current_position is not None else 0.0
        )
        position_side = self._current_position.side if self._current_position is not None else "flat"
        trade_measurement = TradeMeasurement(
            timestamp=self._measurement_timestamp(candle),
            equity=float(self.equity),
            position_size=position_size,
            position_side=position_side,
            drawdown=float(drawdown),
            sharpe_ratio=0.0,
        )
        self.measurement_adapter.record(trade_measurement)
        self._logger.info(
            f"DrawdownStrategy signal={signal} side={side} "
            f"drawdown={float(drawdown):.4f} scale={float(scale):.4f} "
            f"equity={float(self.equity):.4f} size={position_size:.4f}"
        )
