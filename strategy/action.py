"""Actions and positions returned by strategy.ack()."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class Side(IntEnum):
    """Trade direction."""

    SHORT = -1
    FLAT = 0
    LONG = 1

    def __repr__(self) -> str:
        return self.name


@dataclass(slots=True)
class Position:
    """A single position decision produced by a strategy.

    Strategies return a :class:`Position` to describe the desired position
    on the latest bar.  The executor (live or backtest) uses *side* and
    *size* to place or adjust orders.

    Parameters
    ----------
    side : Side
        Desired direction: ``Side.LONG``, ``Side.SHORT``, or ``Side.FLAT``.
    size : float
        Position scale as a fraction of capital (0.0–1.0).
        ``1.0`` means full allocation; ``0.5`` means half.
        Strategies like :class:`~strategy.drawdown.DrawdownPositionSize`
        reduce this based on drawdown.
    price : float | None
        Fill / entry price set by the executor after a simulated or
        real fill.
    stop_loss : float | None
        Optional stop-loss price.  The executor may use this to place
        a protective order.
    take_profit : float | None
        Optional take-profit price.

    Examples
    --------
    ::

        Position(side=Side.LONG, size=0.5)
        Position(side=Side.SHORT, size=0.04, stop_loss=3200.0)
        Position(side=Side.FLAT)            # close position
    """

    side: Side = Side.FLAT
    size: float = 1.0
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    # -- convenience constructors ------------------------------------------

    @classmethod
    def long(cls, size: float = 1.0) -> Position:
        """Create a LONG position."""
        return cls(side=Side.LONG, size=size)

    @classmethod
    def short(cls, size: float = 1.0) -> Position:
        """Create a SHORT position."""
        return cls(side=Side.SHORT, size=size)

    @classmethod
    def flat(cls) -> Position:
        """Create a FLAT (close) position."""
        return cls(side=Side.FLAT, size=0.0)

    # -- conversion helpers ------------------------------------------------

    @classmethod
    def from_raw(cls, position: float) -> Position:
        """Build a Position from a raw numeric position value.

        This bridges the existing convention where strategies produce a
        ``position`` column: sign = direction, magnitude = scale.

        Parameters
        ----------
        position : float
            +1 long, −1 short, 0 flat.  Fractional values (e.g. 0.04)
            are interpreted as scaled long positions.
        """
        if position > 0:
            return cls(side=Side.LONG, size=abs(position))
        elif position < 0:
            return cls(side=Side.SHORT, size=abs(position))
        return cls.flat()

    @property
    def value(self) -> float:
        """Numeric position value (sign × magnitude).

        Compatible with the ``position`` column convention used by the
        existing strategy pipeline and backtester.
        """
        return float(self.side) * self.size

    @property
    def is_flat(self) -> bool:
        return self.side == Side.FLAT

    def __repr__(self) -> str:
        parts = [f"{self.side.name}"]
        if not self.is_flat:
            parts.append(f"size={self.size:.4g}")
        if self.price is not None:
            parts.append(f"price={self.price:.2f}")
        if self.stop_loss is not None:
            parts.append(f"sl={self.stop_loss:.2f}")
        if self.take_profit is not None:
            parts.append(f"tp={self.take_profit:.2f}")
        return f"Position({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class Action:
    """Base class for all strategy actions."""
    pass


class NoAction(Action):
    """No action — hold current state."""

    def __repr__(self) -> str:
        return "NoAction()"


@dataclass(slots=True)
class Open(Action):
    """Open a new position."""

    position: Position
    close_price: float | None = None

    def __repr__(self) -> str:
        if self.close_price is None:
            return f"Open({self.position})"
        return f"Open({self.position}, close_price={self.close_price:.2f})"


@dataclass(slots=True)
class Close(Action):
    """Close the current position."""
    position: Position | None = None
    price: float | None = None

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.position is not None and not self.position.is_flat:
            parts.append(f"size={self.position.size:.4g}")
        if self.price is not None:
            parts.append(f"price={self.price:.2f}")
        if not parts:
            return "Close()"
        return f"Close({', '.join(parts)})"


@dataclass(slots=True)
class Adjust(Action):
    """Adjust the current position (e.g. change size or stops)."""

    position: Position

    def __repr__(self) -> str:
        return f"Adjust({self.position})"
