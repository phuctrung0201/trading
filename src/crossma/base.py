import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.app.config import periods_per_year
from src.app.logger import AppLogger
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position
from src.clickhouse.recorder import Recorder
from src.clickhouse.measurement import TradeMeasurement, TradeEventMeasurement, OpsMeasurement
from src.ema.indicator import EMA


@dataclass
class SignalResult:
    signal: str | None = None
    short_ema: float | None = None
    long_ema: float | None = None


_RECONCILE_INTERVAL_SEC = 300


class BaseStrategy:
    def __init__(
        self,
        exchange: ExchangeAdapter,
        recorder: Recorder,
        logger: AppLogger,
        short_length: int,
        long_length: int,
        steps: str = "1m",
        **kwargs,
    ):
        self.exchange = exchange
        self.recorder = recorder
        self._logger = logger
        self.short = int(short_length)
        self.long = int(long_length)
        self._ema = EMA()
        self._values: list[float] = []
        self._current_position: Position | None = None
        self._exposure_ratio: float = 1.0
        initial_equity = self.exchange.get_equity()
        self._peak_equity: float = initial_equity
        self._returns: list[float] = []
        self._prev_equity: float = initial_equity
        self._periods_per_year: int = periods_per_year(steps)
        self._last_reconcile_time: float = time.monotonic()
        self._last_close_price: float = 0.0

    def _mark_to_market(self):
        equity = self.exchange.get_equity()
        if self._prev_equity > 0:
            period_return = (equity - self._prev_equity) / self._prev_equity
            self._returns.append(period_return)
        self._prev_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def reconcile(self):
        now = time.monotonic()
        if now - self._last_reconcile_time < _RECONCILE_INTERVAL_SEC:
            return
        self._last_reconcile_time = now

        if not hasattr(self.exchange, "fetch_asset"):
            return

        self._logger.info("Reconcile started")
        adapter: Any = self.exchange

        try:
            exchange_equity = adapter.fetch_asset("USDT")
            exchange_position = adapter.fetch_position()
        except Exception as exc:
            self._logger.warning(f"Reconcile fetch failed: {exc}")
            return

        local_has_pos = self._current_position is not None
        exchange_has_pos = exchange_position is not None

        self._logger.info(
            f"Reconcile state "
            f"local_position={local_has_pos} exchange_position={exchange_has_pos}"
        )

        if local_has_pos and not exchange_has_pos:
            assert self._current_position is not None
            self._logger.warning(
                f"Reconcile position gone on exchange "
                f"local_side={self._current_position.side} "
                f"local_size={self._current_position.size:.4f}"
            )
            self._current_position = None
        elif local_has_pos and exchange_has_pos:
            assert self._current_position is not None
            if exchange_position.side != self._current_position.side:
                self._logger.warning(
                    f"Reconcile side mismatch "
                    f"local={self._current_position.side} "
                    f"exchange={exchange_position.side}"
                )
            size_diff = exchange_position.size - self._current_position.size
            self._logger.info(
                f"Reconcile position "
                f"local_side={self._current_position.side} "
                f"exchange_side={exchange_position.side} "
                f"local_size={self._current_position.size:.4f} "
                f"exchange_size={exchange_position.size:.4f} "
                f"size_diff={size_diff:.4f} "
                f"local_entry={self._current_position.price} "
                f"exchange_entry={exchange_position.price}"
            )
            self._current_position.size = exchange_position.size
        elif not local_has_pos and exchange_has_pos:
            self._logger.warning(
                f"Reconcile exchange has position "
                f"side={exchange_position.side} "
                f"size={exchange_position.size:.4f} "
                f"entry={exchange_position.price} "
                f"but local is flat"
            )
        else:
            self._logger.info("Reconcile both flat no position")

        local_equity = self.exchange.get_equity()
        equity_diff = exchange_equity - local_equity
        self._logger.info(
            f"Reconcile equity "
            f"local={local_equity:.4f} exchange={exchange_equity:.4f} "
            f"diff={equity_diff:.4f}"
        )

        position_match = local_has_pos == exchange_has_pos
        correction = None
        if local_has_pos and not exchange_has_pos:
            correction = "cleared local position"
        elif not local_has_pos and exchange_has_pos:
            correction = "exchange has unexpected position"
        elif local_has_pos and exchange_has_pos:
            assert self._current_position is not None
            if exchange_position.side != self._current_position.side:
                correction = (
                    f"side mismatch local={self._current_position.side} "
                    f"exchange={exchange_position.side}"
                )

        adapter.asset = exchange_equity
        self._prev_equity = self.exchange.get_equity()
        if self.exchange.get_equity() > self._peak_equity:
            self._peak_equity = self.exchange.get_equity()

        self._logger.info(
            f"Reconcile applied "
            f"equity={self.exchange.get_equity():.4f} "
            f"peak_equity={self._peak_equity:.4f}"
        )

        now_ns = int(time.time() * 1_000_000_000)
        ops = OpsMeasurement(
            timestamp=now_ns,
            type="reconcile",
            reconcile_equity_diff=equity_diff,
            reconcile_position_match=position_match,
            reconcile_correction=correction,
        )
        self.recorder.record(ops)

    def _calculate_drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self.exchange.get_equity() - self._peak_equity) / self._peak_equity

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if len(self._returns) < 2:
            return 0.0
        mean_return = sum(self._returns) / len(self._returns)
        variance = sum((r - mean_return) ** 2 for r in self._returns) / len(self._returns)
        std_dev = variance ** 0.5
        if std_dev == 0:
            return 0.0
        sharpe = (mean_return - risk_free_rate) / std_dev
        return sharpe * (self._periods_per_year ** 0.5)

    def warmup(self, trade: MarketTrade):
        close_value = getattr(trade, "close", None)
        if close_value is not None:
            self._values.append(float(close_value))

    def _signal(self, trade: MarketTrade) -> SignalResult:
        close_value = getattr(trade, "close", None)
        if close_value is None:
            return SignalResult()

        self._values.append(float(close_value))
        if len(self._values) < self.long:
            return SignalResult()

        short_values = self._values[-self.short:]
        long_values = self._values[-self.long:]
        short_ema = self._ema.calculate(short_values)
        long_ema = self._ema.calculate(long_values)

        if short_ema is None or long_ema is None:
            return SignalResult()

        signal = self.compute_signal(short_ema, long_ema)
        return SignalResult(signal=signal, short_ema=short_ema, long_ema=long_ema)

    @abstractmethod
    def compute_signal(self, short_ema: float, long_ema: float) -> str | None:
        raise NotImplementedError

    def _emit_event(
        self,
        trade: MarketTrade,
        event: str,
        signal_result: SignalResult | None = None,
        fill_price: float | None = None,
        pnl: float | None = None,
        signal: str | None = None,
        reason: str | None = None,
    ):
        position_size = (
            float(self._current_position.size) if self._current_position is not None else 0.0
        )
        position_side = self._current_position.side if self._current_position is not None else "flat"
        if position_side == "sell":
            position_size = -position_size
        event_measurement = TradeEventMeasurement(
            timestamp=self._measurement_timestamp(trade),
            event=event,
            equity=self.exchange.get_equity(),
            close_price=float(getattr(trade, "close", 0) or 0),
            position_size=position_size,
            position_side=position_side,
            drawdown=self._calculate_drawdown(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            short_ema=signal_result.short_ema if signal_result else None,
            long_ema=signal_result.long_ema if signal_result else None,
            exposure_ratio=self._exposure_ratio,
            fill_price=fill_price,
            pnl=pnl,
            signal=signal,
            reason=reason,
        )
        self.recorder.record(event_measurement)

    def _measurement_timestamp(self, trade: MarketTrade) -> int | None:
        raw = getattr(trade, "timestamp", None)
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

    def _emit_trade_measurement(self, trade: MarketTrade, result: SignalResult,
                                drawdown: float | None = None):
        close_price = float(getattr(trade, "close", 0) or 0)
        self._last_close_price = close_price
        position_size = (
            float(self._current_position.size) if self._current_position is not None else 0.0
        )
        position_side = self._current_position.side if self._current_position is not None else "flat"
        if position_side == "sell":
            position_size = -position_size
        dd = drawdown if drawdown is not None else self._calculate_drawdown()
        measurement = TradeMeasurement(
            timestamp=self._measurement_timestamp(trade),
            equity=self.exchange.get_equity(),
            position_size=position_size,
            position_side=position_side,
            drawdown=dd,
            sharpe_ratio=self._calculate_sharpe_ratio(),
            short_ema=result.short_ema,
            long_ema=result.long_ema,
            close_price=close_price,
            exposure_ratio=self._exposure_ratio,
        )
        self.recorder.record(measurement)

    def _execute(self, trade: MarketTrade, result: SignalResult):
        """Common open/close execution for signal-driven strategies."""
        if result.signal is None:
            return

        side = "buy" if result.signal == "LONG" else "sell"
        if self._current_position is not None and self._current_position.side != side:
            close_pnl = self.exchange.unrealized_pnl()
            self._logger.info(
                f"Closing side={self._current_position.side} "
                f"size={self._current_position.size:.4f}"
            )
            self.exchange.close(self._current_position)
            self._emit_event(
                trade, "close", signal_result=result,
                fill_price=self._last_close_price,
                pnl=close_pnl,
                signal=result.signal,
                reason=f"signal flip to {result.signal}",
            )
            self._current_position = None
            self._exposure_ratio = 1.0

        if self._current_position is None:
            equity = self.exchange.get_equity()
            position = Position(side=side, size=equity)
            open_result = self.exchange.open(position)
            if open_result.success:
                self._current_position = open_result.position or position
                self._logger.info(
                    f"Open signal={result.signal} side={side} "
                    f"equity={equity:.4f} size={self._current_position.size:.4f} "
                    f"fill_price={self._current_position.price}"
                )
                self._emit_event(
                    trade, "open", signal_result=result,
                    fill_price=self._current_position.price,
                    signal=result.signal,
                    reason=f"EMA crossover {result.signal}",
                )
            else:
                self._logger.error(
                    f"Open failed signal={result.signal} side={side} "
                    f"message={open_result.message}"
                )
                self._emit_event(
                    trade, "error", signal_result=result,
                    reason=f"open failed: {open_result.message}",
                )

    @abstractmethod
    def ack(self, trade: MarketTrade):
        raise NotImplementedError
