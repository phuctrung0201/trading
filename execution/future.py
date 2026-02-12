"""Futures trading executor for live strategy signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from source.okx import Candle

if TYPE_CHECKING:
    from source.okx import Client


# ---------------------------------------------------------------------------
# Result – returned by evaluate() on position changes
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Summary of a position change produced by :func:`evaluate`.

    Attributes
    ----------
    time : str
        Timestamp of the bar that triggered the change.
    price : float
        Close price at which the change occurred.
    position : str
        New position label (``"LONG"``, ``"SHORT"``, or ``"FLAT"``).
    equity : float
        Account equity after the change.
    opened : str | None
        Side that was opened (``"LONG"`` or ``"SHORT"``), or ``None``.
    closed : str | None
        Side that was closed, or ``None``.
    pnl : float | None
        Realised PnL from closing the previous position, or ``None``.
    value : float | None
        Notional value allocated to the new position, or ``None``.
    """

    time: str
    price: float
    position: str
    equity: float
    opened: str | None = None
    closed: str | None = None
    pnl: float | None = None
    value: float | None = None
    indicators: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [self.time]
        if self.closed:
            parts.append(f"CLOSE={self.closed}")
        if self.pnl is not None:
            parts.append(f"PnL={self.pnl:+.2f}")
        if self.opened:
            parts.append(f"OPEN={self.opened}")
        if self.value is not None:
            parts.append(f"V={self.value:.2f}")
        parts.append(f"P={self.price:.2f}")
        parts.append(f"E={self.equity:.2f}")
        for k, v in self.indicators.items():
            parts.append(f"{k}={v:.4f}")
        parts.append(f"[{self.position}]")
        return "  ".join(parts)


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

        print(
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


# ---------------------------------------------------------------------------
# action – run strategies and execute paper trades
# ---------------------------------------------------------------------------

_POS_LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def evaluate(
    context: NewContext,
    strategies: list,
    okx: "Client | None" = None,
) -> Result | None:
    """Run the strategy pipeline on the current OHLCV data and trade futures.

    The last strategy in *strategies* is the base signal generator;
    earlier entries act as overlays (same convention as the backtester).

    Position sizing is determined by the entry overlay: the magnitude of
    the ``position`` signal is used as the fraction of capital to allocate.

    A trade is printed whenever the target position differs from the
    current position.  If *okx* is provided, real market orders are
    placed on OKX.

    Parameters
    ----------
    context : NewContext
        Futures trading context (updated via :meth:`NewContext.update`).
    strategies : list
        Strategy instances that implement ``generate_signals(df)``.
    okx : Client, optional
        OKX client instance for order execution.

    Returns
    -------
    Result | None
        A :class:`Result` when a position change occurs, ``None`` otherwise.
    """
    # Configure leverage on the exchange once
    if okx and context.instrument and not context._leverage_set:
        okx.set_leverage(
            instrument=context.instrument,
            leverage=context.leverage,
            margin_mode=context.margin_mode,
        )
        context._leverage_set = True

    # Need enough bars for the strategy to produce a signal
    if len(context.ohlc) < 2:
        return None

    # Run strategy pipeline (last = base, earlier = overlays)
    signals = strategies[-1].generate_signals(context.ohlc)
    for strategy in strategies[:-1]:
        signals = strategy.generate_signals(signals)

    if "position" not in signals.columns:
        return None

    raw_position = float(signals["position"].iloc[-1])
    price = float(context.ohlc["close"].iloc[-1])
    ts = context.ohlc.index[-1]

    # Direction: +1 long, -1 short, 0 flat.  Magnitude = scale factor.
    if raw_position > 0:
        new_direction = 1
    elif raw_position < 0:
        new_direction = -1
    else:
        new_direction = 0
    position_scale = abs(raw_position) if abs(raw_position) > 0 else 1.0

    if new_direction == context.position:
        return None

    instrument = context.instrument
    position_value = context.capital * position_scale

    closed_label: str | None = None
    pnl: float | None = None
    opened_label: str | None = None
    open_value: float | None = None

    # -- close existing position ------------------------------------------
    if context.position != 0 and context.entry_price is not None:
        if context.position == 1:
            pnl = (price - context.entry_price) / context.entry_price * context.capital * context.entry_scale
        else:
            pnl = (context.entry_price - price) / context.entry_price * context.capital * context.entry_scale

        context.capital += pnl
        closed_label = _POS_LABEL[context.position]
        context.trades.append({
            "time": str(ts),
            "action": "close",
            "side": closed_label,
            "price": price,
            "pnl": pnl,
            "equity": context.capital,
        })

        # Execute close on OKX
        if okx and instrument:
            _execute_close(okx, instrument, context)

    # -- open new position ------------------------------------------------
    if new_direction != 0:
        context.entry_price = price
        context.entry_scale = position_scale
        opened_label = _POS_LABEL[new_direction]
        open_value = position_value
        context.trades.append({
            "time": str(ts),
            "action": "open",
            "side": opened_label,
            "price": price,
            "equity": context.capital,
        })

        # Execute open on OKX
        if okx and instrument:
            _execute_open(okx, instrument, context, new_direction, position_value)
    else:
        context.entry_price = None

    context.position = new_direction

    # Collect strategy indicator values from the last row
    indicator_cols = {"ma_fast", "ma_slow", "ema_short", "ema_long", "macd", "macd_signal", "macd_hist",
                      "drawdown_pct", "drawup_pct"}
    indicators = {}
    for col in indicator_cols:
        if col in signals.columns:
            val = signals[col].iloc[-1]
            if pd.notna(val):
                indicators[col] = float(val)

    return Result(
        time=str(ts.tz_localize(None) if hasattr(ts, 'tz_localize') and ts.tzinfo else ts),
        price=price,
        position=_POS_LABEL[new_direction],
        equity=context.capital,
        opened=opened_label,
        closed=closed_label,
        pnl=pnl,
        value=open_value,
        indicators=indicators,
    )


# ---------------------------------------------------------------------------
# OKX order execution helpers
# ---------------------------------------------------------------------------

def _execute_close(client: "Client", instrument: str, context: NewContext) -> None:
    """Close the current position on OKX."""
    try:
        if context.holding_size == "0":
            return

        result = client.close_position(
            instrument=instrument,
            margin_mode="cross",
        )
        context.holding_size = "0"
    except Exception as e:
        print(f"         OKX close position failed: {e}")


def _execute_open(
    client: "Client",
    instrument: str,
    context: NewContext,
    position: int,
    position_value: float,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> None:
    """Open a new futures position on OKX with retry on failure.

    Parameters
    ----------
    position_value : float
        Notional value in quote currency (USDT) to allocate to this position.
    max_retries : int
        Maximum number of attempts before giving up.
    retry_delay : float
        Seconds to wait between retries.
    """
    import time

    side = "buy" if position == 1 else "sell"
    # For SWAP contracts, size is in number of contracts
    # ETH-USDT-SWAP: 1 contract = 0.01 ETH
    price = context.entry_price or 1
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
            context.holding_size = size
            return
        except Exception as e:
            print(f"         OKX order failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)

    print(f"         OKX order gave up after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Future – OO wrapper matching the Backtester interface
# ---------------------------------------------------------------------------


class Future:
    """Object-oriented futures executor with a streaming candle-by-candle API.

    Mirrors the :class:`~execution.backtester.Backtester` interface so that
    live-trading scripts follow the same pattern as backtest scripts.

    Usage
    -----
    ::

        executor = Future(cap=capital, instrument="ETH-USDT-SWAP",
                          leverage=10, okx=client, ohlc=prices,
                          strategy=DrawdownPositionSize(signal=MACross(short=5, long=10), ...))

        for candle in channel:
            executor.ack(candle)
            if candle.confirm:
                result = executor.exec()

    Parameters
    ----------
    cap : float
        Starting capital in quote currency.
    instrument : str
        Exchange instrument ID (e.g. ``"ETH-USDT-SWAP"``).
    leverage : int
        Leverage multiplier.
    strategy
        Strategy / risk-management overlay (must contain a signal).
    okx : Client, optional
        OKX client for order execution.
    ohlc : pd.DataFrame, optional
        Pre-loaded OHLCV data.
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

    # -- streaming API ------------------------------------------------------

    def ack(self, candle: Candle) -> None:
        """Add or update a candle in the context."""
        self._context.update(candle)

    def exec(self) -> Result | None:
        """Run the strategy pipeline and execute trades.

        Returns
        -------
        Result | None
            A :class:`Result` when a position change occurs, ``None`` otherwise.
        """
        strategies = [self._strategy]

        return evaluate(
            context=self._context,
            strategies=strategies,
            okx=self._okx,
        )
