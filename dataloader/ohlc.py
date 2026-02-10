"""Utilities for loading OHLCV data from CSV files."""

import pandas as pd


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
