"""Utilities for loading OHLCV data from CSV files."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class Candle:
    """A single OHLCV bar.

    This is the generic, client-agnostic candle used by the strategy
    layer.  Exchange-specific candle types (e.g. ``client.okx.Candle``)
    can be converted via :meth:`from_okx` or by constructing directly.

    Parameters
    ----------
    timestamp : str
        Bar timestamp (ISO-formatted string, e.g. ``"2025-01-15 12:00:00"``).
    open : float
        Opening price.
    high : float
        Highest price.
    low : float
        Lowest price.
    close : float
        Closing price.
    volume : float
        Trade volume.
    timestamp_ns : int | None
        Optional Unix timestamp in nanoseconds for fast downstream writes.
    """

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp_ns: int | None = None

    # -- convenience constructors ------------------------------------------

    @classmethod
    def from_series(cls, series: pd.Series) -> Candle:
        """Build a Candle from a DataFrame row (``pd.Series``).

        The series index is expected to contain ``open``, ``high``,
        ``low``, ``close``, ``volume``.  The ``.name`` attribute is
        used as the timestamp.
        """
        return cls(
            timestamp=str(series.name),
            open=float(series["open"]),
            high=float(series["high"]),
            low=float(series["low"]),
            close=float(series["close"]),
            volume=float(series["volume"]),
            timestamp_ns=int(series.name.value) if hasattr(series.name, "value") else None,
        )

    @classmethod
    def from_dict(cls, data: dict) -> Candle:
        """Build a Candle from a dictionary."""
        return cls(
            timestamp=str(data["timestamp"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            timestamp_ns=int(data["timestamp_ns"]) if "timestamp_ns" in data and data["timestamp_ns"] is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"{self.timestamp}  O={self.open:.4f}  H={self.high:.4f}  "
            f"L={self.low:.4f}  C={self.close:.4f}  V={self.volume:.2f}"
        )


def csv(path: str) -> pd.DataFrame:
    """Read an OHLCV CSV file and return a DataFrame.

    The CSV is expected to have columns:
    ``timestamp, open, high, low, close, volume``.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and columns: open, high, low, close, volume.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.strip().lower() for c in df.columns]

    for required in ("open", "high", "low", "close", "volume"):
        if required not in df.columns:
            raise ValueError(f"Missing required column: '{required}'")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])

    return df
