"""Futures trading executor for live strategy signals."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING

import pandas as pd

from dataloader.ohlc import Candle
from executor.noaction import NoActionExecution
from logger import log
from monitor.measurement import TradeMeasurement
from strategy.action import Action, NoAction, Open, Close, Adjust

if TYPE_CHECKING:
    from client.influxdb import InfluxClient
    from client.okx import Client


class ExecutionError(Exception):
    """Raised when exchange execution fails."""


class OkxExcutor(NoActionExecution):
    """Futures executor with a streaming candle-by-candle API.

    Extends :class:`NoActionExecution` so all executor types share the
    same ``ack(candle)`` interface.

    On each call to :meth:`ack` it feeds the candle to the strategy
    and carries out the resulting action.

    Parameters
    ----------
    instrument : str
        Exchange instrument ID (e.g. ``"ETH-USDT-SWAP"``).
    leverage : int
        Leverage multiplier.
    strategy
        Strategy instance whose ``ack`` method returns an Action.
    okx : Client, optional
        OKX client for order execution.
    ohlc : pd.DataFrame, optional
        Pre-loaded OHLCV data fed to the strategy on init so it has
        enough history to generate signals immediately.
    """

    def __init__(
        self,
        instrument: str,
        leverage: int,
        strategy,
        okx: "Client | None" = None,
        ohlc: pd.DataFrame | None = None,
        influx_client: "InfluxClient | None" = None,
        session_id: str | None = None,
    ) -> None:
        self._okx = okx
        self._strategy = strategy
        self._influx_client = influx_client
        self._session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._trade_measurement: TradeMeasurement | None = None
        if self._influx_client is not None:
            self._trade_measurement = TradeMeasurement(session_id=self._session_id)
        self._instrument = instrument
        self._leverage = leverage
        self._margin_mode = "cross"
        self._leverage_set = False
        starting_equity = float(self._strategy.current_equity())

        log.info(
            f"Future executor: equity={starting_equity:.2f}, "
            f"instrument={self._instrument}, leverage={self._leverage}x, "
            f"preloaded={len(ohlc) if ohlc is not None else 0} bars"
        )

        # Warm up strategy with pre-loaded historical data
        if ohlc is not None and len(ohlc) > 0:
            for _, row in ohlc.iterrows():
                c = Candle.from_series(row)
                action = self._strategy.ack(c)
                self._strategy.confirm(action)

    def ack(self, candle: Candle) -> None:
        action = self._strategy.ack(candle)
        self.execute(action, candle)
        self._strategy.confirm(action)
        position = self._strategy.current_position()
        self._write_trade_influx(
            timestamp_ns=time.time_ns(),
            equity=float(self._strategy.current_equity()),
            position_side=position.side.name,
            position_size=float(position.size),
        )

    def execute(self, action: Action, candle: Candle) -> None:
        """Execute a strategy action against the current candle."""
        okx = self._okx

        # Configure leverage on the exchange once
        if okx and self._instrument and not self._leverage_set:
            okx.set_leverage(
                instrument=self._instrument,
                leverage=self._leverage,
                margin_mode=self._margin_mode,
            )
            self._leverage_set = True

        if isinstance(action, NoAction):
            return

        price = float(candle.close)
        instrument = self._instrument

        if isinstance(action, Close):
            try:
                close_size = action.position.size if action.position is not None else 0.0
                close_result = self._close_position(
                    price=price,
                    instrument=instrument,
                    close_size=close_size,
                )
                if close_result is not None:
                    action.price = close_result
            except ExecutionError:
                return
            return

        if isinstance(action, Open):
            prev_position = self._strategy.current_position()
            if not prev_position.is_flat:
                try:
                    close_result = self._close_position(
                        price=price,
                        instrument=instrument,
                        close_size=prev_position.size,
                    )
                    if close_result is not None:
                        action.close_price = close_result
                except ExecutionError:
                    return

            pos = action.position
            equity = float(self._strategy.current_equity())
            position_value = equity * pos.size
            entry_price = price

            if okx and instrument:
                try:
                    entry_price = self._execute_open(
                        instrument=instrument,
                        position=int(pos.side),
                        position_value=position_value,
                        reference_price=price,
                    )
                except ExecutionError:
                    return

            # Report effective fill price back to strategy in confirm(action)
            pos.price = entry_price
            return

        if isinstance(action, Adjust):
            return

    def _close_position(
        self,
        price: float,
        instrument: str,
        close_size: float,
    ) -> float | None:
        """Close current exchange position."""
        if close_size <= 0:
            return None

        if self._okx and instrument:
            self._execute_close(instrument=instrument)

        return price

    def _execute_close(self, instrument: str) -> None:
        """Close current position on OKX."""
        client = self._okx
        if client is None:
            raise ExecutionError("okx client missing")
        try:
            client.close_position(instrument=instrument, margin_mode=self._margin_mode)
        except Exception as e:
            log.error(f"OKX close position failed: {e}")
            raise ExecutionError("close position failed") from e

    def _execute_open(
        self,
        instrument: str,
        position: int,
        position_value: float,
        reference_price: float,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> float:
        """Open a new futures position on OKX with retry on failure."""
        import time

        client = self._okx
        if client is None:
            raise ExecutionError("okx client missing")

        side = "buy" if position == 1 else "sell"
        price = reference_price or 1
        contracts = int(position_value / price / 0.01) or 1
        size = str(contracts)

        for attempt in range(1, max_retries + 1):
            try:
                order = client.place_order(
                    instrument=instrument,
                    side=side,
                    size=size,
                    order_type="market",
                    trade_mode="cross",
                )
                order_price = self._safe_float(order.price)
                if order_price is None or order_price <= 0:
                    order_price = reference_price
                return order_price
            except Exception as e:
                log.error(f"OKX order failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)

        log.error(f"OKX order gave up after {max_retries} attempts")
        raise ExecutionError("open position failed")

    @staticmethod
    def _safe_float(value: str | float | int | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_trade_influx(
        self,
        *,
        timestamp_ns: int,
        equity: float,
        position_side: str,
        position_size: float,
    ) -> None:
        if self._influx_client is None or self._trade_measurement is None:
            return
        value = self._trade_measurement.values(
            timestamp_ns=timestamp_ns,
            equity=equity,
            position_side=position_side,
            position_size=position_size,
        )
        self._influx_client.write(value)
