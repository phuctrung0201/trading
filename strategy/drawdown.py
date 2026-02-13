"""Drawdown-aware strategy that scales position size based on drawdown."""

from __future__ import annotations

from collections import deque

from dataloader.ohlc import Candle
from strategy.action import Action, NoAction, Open, Close, Adjust, Position
from strategy.noaction import NoActionStrategy


class DrawdownPositionSize(NoActionStrategy):
    """Scale position size by drawdown level.

    Position size is scaled according to *size* thresholds based
    on the current drawdown level.

    Parameters
    ----------
    signals : list
        Signal instances that implement ``step(close)`` returning a
        position value.
    size : dict[float, float]
        Mapping of *drawdown-fraction → position-scale*.  Keys are
        drawdown levels as decimals (e.g. ``0.04`` means 4 %) and
        values are the multiplier applied to the position when the
        drawdown reaches that level.  Higher drawdown thresholds take
        priority.

        Example::

            {
                0: 0.5,       # no drawdown      → 50 % size
                0.04: 0.04,   # at 4 % drawdown  → 4 % size
                0.06: 0.02,   # at 6 % drawdown  → 2 % size
            }
    drawdown_window : int
        Rolling lookback window for the equity peak.  Only the last
        *window* equity values are considered when determining the
        peak, so old highs expire over time.
    """

    def __init__(
        self,
        signals: list,
        size: dict[float, float],
        drawdown_window: int,
        equity: float,
    ) -> None:
        super().__init__(equity=equity)
        self.signals = signals
        self.drawdown_window = drawdown_window
        self.last_position: Position = Position.flat()
        # Sort descending so the highest (most severe) threshold matches first.
        self.thresh_hold: dict[float, float] = {
            float(k): v for k, v in sorted(size.items(), reverse=True)
        }

        # -- incremental state -----------------------------------------------
        self._prev_close: float | None = None
        self._scaled_position: float = 0.0
        self._equity_window: deque[float] = deque([equity], maxlen=drawdown_window)


    def __repr__(self) -> str:
        return (
            f"DrawdownPositionSize(signals={len(self.signals)}, "
            f"thresh_hold={self.thresh_hold})"
        )

    def ack(self, candle: Candle) -> Action:
        """Process candle and decide the next action."""
        close = float(candle.close)

        # Step signals to get current positions
        for s in self.signals:
            s.step(close)

        # Update equity & drawdown from candle
        if self._prev_close is not None and self._prev_close != 0:
            bar_ret = (close - self._prev_close) / self._prev_close
            self._equity *= 1.0 + self._scaled_position * bar_ret
            self._equity_window.append(self._equity)
        self._prev_close = close

        raw_position = float(self.signals[0]._position)

        # Scale by current drawdown
        peak = max(self._equity_window)
        drawdown_pct = (peak - self._equity) / peak if peak > 0 else 0.0

        scale = 1.0
        for dd_level, dd_scale in self.thresh_hold.items():
            if drawdown_pct >= dd_level:
                scale = dd_scale
                break

        scaled = raw_position * scale

        # Determine action
        prev = self.last_position
        curr = Position.from_raw(scaled)

        if prev.side == curr.side and prev.size == curr.size:
            return NoAction()
        if prev.is_flat and not curr.is_flat:
            return Open(position=curr)
        if not prev.is_flat and curr.is_flat:
            return Close(position=prev)
        if prev.side != curr.side:
            return Open(position=curr)
        return Adjust(position=curr)

    def confirm(self, action: Action) -> None:
        """Update position tracking with the actual action."""
        if isinstance(action, (Open, Adjust)):
            self._scaled_position = action.position.value
            self.last_position = action.position
        elif isinstance(action, Close):
            self._scaled_position = 0.0
            self.last_position = Position.flat()

    def current_position(self) -> Position:
        return self.last_position

