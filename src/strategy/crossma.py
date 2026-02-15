from datetime import datetime, timezone

from src.app.logger import AppLogger
from src.client.ohclv import OHCLV
from src.indicator.ema import EMA
from src.measurement.trade import TradeMeasurement
from src.strategy.adapter import ExchangeAdapter, MeasurementAdapter, Position
from src.strategy.noaction import NoActionStrategy


class CrossMAStrategy(NoActionStrategy):
    def init_equity(self, exchange_adapter):
        try:
            return float(exchange_adapter.get_asset("USDT"))
        except Exception:
            return 0.0

    def init_short_length(self, length):
        self.short = int(length)

    def init_long_length(self, length):
        self.long = int(length)

    def __init__(
        self,
        exchange_adapter: ExchangeAdapter,
        measurement_adapter: MeasurementAdapter,
        app_logger: AppLogger,
        periods_per_year: int = 525600,
    ):
        super().__init__(app_logger=app_logger)
        self.exchange_adapter: ExchangeAdapter = exchange_adapter
        self.measurement_adapter = measurement_adapter
        self.init_short_length(7)
        self.init_long_length(21)
        self.equity: float = self.init_equity(exchange_adapter)
        self._ema: EMA = EMA()
        self._values: list[float] = []
        self._current_position: Position | None = None
        self._last_close: float | None = None
        self._peak_equity: float = self.equity
        self._returns: list[float] = []
        self._prev_equity: float = self.equity
        self._periods_per_year: int = periods_per_year

    def _mark_to_market(self, candle: OHCLV):
        close_value = getattr(candle, "close", None)
        if close_value is None:
            return

        close_price = float(close_value)
        if (
            self._last_close is not None
            and self._last_close > 0
            and self._current_position is not None
            and self._current_position.size > 0
        ):
            bar_return = (close_price - self._last_close) / self._last_close
            direction = 1.0 if self._current_position.side == "buy" else -1.0
            pnl = self._current_position.size * bar_return * direction
            self.equity = max(0.0, self.equity + pnl)
            self._current_position.size = self.equity

        self._last_close = close_price

        if self._prev_equity > 0:
            period_return = (self.equity - self._prev_equity) / self._prev_equity
            self._returns.append(period_return)
        self._prev_equity = self.equity

        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

    def _calculate_drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self.equity - self._peak_equity) / self._peak_equity

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self._returns) < 2:
            return 0.0
        mean_return = sum(self._returns) / len(self._returns)
        variance = sum((r - mean_return) ** 2 for r in self._returns) / len(self._returns)
        std_dev = variance ** 0.5
        if std_dev == 0:
            return 0.0
        sharpe = (mean_return - risk_free_rate) / std_dev
        return sharpe * (self._periods_per_year ** 0.5)

    def is_long(self, short_ema, long_ema):
        return short_ema > long_ema

    def is_short(self, short_ema, long_ema):
        return short_ema < long_ema

    def _signal(self, candle: OHCLV):
        close_value = getattr(candle, "close", None)
        if close_value is None:
            return None

        self._values.append(float(close_value))
        if len(self._values) < self.long:
            return None

        short_values = self._values[-self.short :]
        long_values = self._values[-self.long :]
        short_ema = self._ema.calculate(short_values)
        long_ema = self._ema.calculate(long_values)

        if short_ema is None or long_ema is None:
            return None
        if self.is_long(short_ema, long_ema):
            return "LONG"
        if self.is_short(short_ema, long_ema):
            return "SHORT"
        return None

    def _measurement_timestamp(self, candle: OHCLV) -> int | None:
        raw = getattr(candle, "timestamp", None)
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return int(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if text.endswith("Z"):
                text = text.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        return None

    def ack(self, candle: OHCLV):
        timestamp = getattr(candle, "timestamp", None)
        self._logger.info(f"CrossMAStrategy ack timestamp={timestamp}")
        self._mark_to_market(candle)
        signal = self._signal(candle)
        if signal is None:
            self._logger.info("CrossMAStrategy no signal")
            return

        side = "buy" if signal == "LONG" else "sell"
        if self._current_position is not None and self._current_position.side != side:
            self.exchange_adapter.close(self._current_position)
            self._current_position = None

        if self._current_position is None:
            position = Position(side=side, size=self.equity)
            result = self.exchange_adapter.open(position)
            if result.success:
                self._current_position = result.position or position

        position_size = (
            float(self._current_position.size) if self._current_position is not None else 0.0
        )
        position_side = self._current_position.side if self._current_position is not None else "flat"
        drawdown = self._calculate_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        trade_measurement = TradeMeasurement(
            timestamp=self._measurement_timestamp(candle),
            equity=float(self.equity),
            position_size=position_size,
            position_side=position_side,
            drawdown=drawdown,
            sharpe_ratio=sharpe_ratio,
        )
        self.measurement_adapter.record(trade_measurement)
        self._logger.info(
            f"CrossMAStrategy signal={signal} side={side} "
            f"equity={float(self.equity):.4f} size={position_size:.4f}"
        )
