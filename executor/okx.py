"""Futures trading executor for live strategy signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from dataloader.ohlc import Candle
from executor.noaction import NoActionExecution
from logger import log
from strategy.action import Action, NoAction, Open, Close, Adjust, Side

if TYPE_CHECKING:
    from client.okx import Client


class ExecutionError(Exception):
    """Raised when exchange execution fails."""


# ---------------------------------------------------------------------------
# Context – accumulates candles and tracks paper-trade state
# ---------------------------------------------------------------------------

@dataclass
class NewContext:
    """Futures trading context that holds OHLCV history and position state.

    Parameters
    ----------
    capital : float
        Starting capital in quote currency (e.g. USDT).
    ohlc : pd.DataFrame, optional
        Pre-loaded OHLCV DataFrame (e.g. from historical candles).
        If provided, the context starts with this data so strategies
        have enough bars to generate signals immediately.
    instrument : str
        Instrument ID for order execution, e.g. ``"ETH-USDT-SWAP"``.
    leverage : int
        Leverage value, e.g. ``10``.
    margin_mode : str
        ``"cross"`` or ``"isolated"``.
    """

    capital: float = 1000.0
    ohlc: pd.DataFrame | None = None
    instrument: str = ""
    leverage: int = 10
    margin_mode: str = "cross"

    # -- position tracking --
    position: int = field(default=0, init=False)       # +1 long, -1 short, 0 flat
    entry_price: float | None = field(default=None, init=False)
    entry_scale: float = field(default=1.0, init=False)  # position scale at entry
    holding_size: str = field(default="0", init=False)  # actual size on exchange

    # -- accounting --
    initial_capital: float = field(default=0.0, init=False)
    trades: list = field(default_factory=list, init=False)

    # -- internal state --
    _last_ts: str | None = field(default=None, init=False, repr=False)
    _leverage_set: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.initial_capital = self.capital
        if self.ohlc is None:
            self.ohlc = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
            )
            self.ohlc.index.name = "timestamp"
        else:
            # Use a copy so the caller's DataFrame is not mutated
            self.ohlc = self.ohlc.copy()
            if len(self.ohlc) > 0:
                self._last_ts = str(self.ohlc.index[-1])

        log.info(
            f"Future context: capital={self.capital:.2f}, "
            f"instrument={self.instrument}, leverage={self.leverage}x, "
            f"preloaded={len(self.ohlc)} bars"
        )

    # ------------------------------------------------------------------

    def update(self, candle: Candle) -> None:
        """Add or update a candle in the OHLCV DataFrame.

        - If the candle has the **same** timestamp as the latest row, the
          row is updated in place (intra-bar tick).
        - If it has a **new** timestamp, a new row is appended.
        """
        ts = pd.Timestamp(candle.timestamp, tz="UTC")
        row = {
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume),
        }

        if self._last_ts == candle.timestamp:
            # Same candle – update in place
            self.ohlc.loc[ts] = row
        else:
            # New candle
            self._last_ts = candle.timestamp
            new_row = pd.DataFrame(
                [row],
                index=pd.DatetimeIndex([ts], name="timestamp"),
            )
            self.ohlc = pd.concat([self.ohlc, new_row])


class OkxExcutor(NoActionExecution):
    """Futures executor with a streaming candle-by-candle API.

    Extends :class:`NoActionExecution` so all executor types share the
    same ``ack(candle)`` interface.

    On each call to :meth:`ack` it feeds the candle to the strategy
    and carries out the resulting action.

    Parameters
    ----------
    cap : float
        Starting capital in quote currency.
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
        cap: float,
        instrument: str,
        leverage: int,
        strategy,
        okx: "Client | None" = None,
        ohlc: pd.DataFrame | None = None,
    ) -> None:
        self._okx = okx
        self._strategy = strategy
        self._context = NewContext(
            capital=cap,
            ohlc=ohlc,
            instrument=instrument,
            leverage=leverage,
        )

        # Warm up strategy with pre-loaded historical data
        if ohlc is not None and len(ohlc) > 0:
            for _, row in ohlc.iterrows():
                c = Candle.from_series(row)
                action = self._strategy.ack(c)
                self._strategy.confirm(action)

    def ack(self, candle: Candle) -> None:
        self._context.update(candle)
        action = self._strategy.ack(candle)
        self.execute(action)
        self._strategy.confirm(action)

    def execute(self, action: Action) -> None:
        """Execute a strategy action against current context."""
        context = self._context
        okx = self._okx

        # Configure leverage on the exchange once
        if okx and context.instrument and not context._leverage_set:
            okx.set_leverage(
                instrument=context.instrument,
                leverage=context.leverage,
                margin_mode=context.margin_mode,
            )
            context._leverage_set = True

        if isinstance(action, NoAction):
            return

        price = float(context.ohlc["close"].iloc[-1])
        ts = context.ohlc.index[-1]
        instrument = context.instrument

        if isinstance(action, Close):
            try:
                self._close_position(price=price, ts=ts, instrument=instrument)
            except ExecutionError:
                return
            return

        if isinstance(action, Open):
            try:
                self._close_position(price=price, ts=ts, instrument=instrument)
            except ExecutionError:
                return

            pos = action.position
            position_value = context.capital * pos.size
            entry_price = price
            holding_size = context.holding_size

            if okx and instrument:
                try:
                    holding_size, entry_price = self._execute_open(
                        instrument=instrument,
                        position=int(pos.side),
                        position_value=position_value,
                        reference_price=price,
                    )
                except ExecutionError:
                    return

            context.entry_price = entry_price
            context.entry_scale = pos.size
            context.position = int(pos.side)
            context.holding_size = holding_size
            context.trades.append({
                "time": str(ts),
                "action": "open",
                "side": pos.side.name,
                "price": entry_price,
                "equity": context.capital,
            })
            return

        if isinstance(action, Adjust):
            pos = action.position
            context.entry_scale = pos.size
            context.position = int(pos.side)

    def _close_position(self, price: float, ts: pd.Timestamp, instrument: str) -> None:
        """Close local/exchange position and realise PnL."""
        context = self._context
        if context.position == 0 or context.entry_price is None:
            return

        if self._okx and instrument:
            self._execute_close(instrument=instrument)

        if context.position == 1:
            pnl = (price - context.entry_price) / context.entry_price * context.capital * context.entry_scale
        else:
            pnl = (context.entry_price - price) / context.entry_price * context.capital * context.entry_scale

        context.capital += pnl
        context.trades.append({
            "time": str(ts),
            "action": "close",
            "side": Side(context.position).name,
            "price": price,
            "pnl": pnl,
            "equity": context.capital,
        })
        context.entry_price = None
        context.position = 0
        context.holding_size = "0"

    def _execute_close(self, instrument: str) -> None:
        """Close current position on OKX."""
        context = self._context
        client = self._okx
        if client is None:
            raise ExecutionError("okx client missing")
        if context.holding_size == "0":
            return
        try:
            client.close_position(instrument=instrument, margin_mode="cross")
        except Exception as e:
            log.error(f"OKX close position failed: {e}")
            raise ExecutionError("close position failed") from e
        context.holding_size = "0"

    def _execute_open(
        self,
        instrument: str,
        position: int,
        position_value: float,
        reference_price: float,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> tuple[str, float]:
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
                return size, order_price
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
