"""Utilities for loading order book data from CSV files."""

import pandas as pd


def csv(path: str) -> pd.DataFrame:
    """Read an order book CSV file and return a DataFrame.

    The CSV is expected to have columns:
    ``timestamp, side, price, quantity``.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, side, price, quantity.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = [c.strip().lower() for c in df.columns]

    for required in ("timestamp", "side", "price", "quantity"):
        if required not in df.columns:
            raise ValueError(f"Missing required column: '{required}'")

    df["price"] = pd.to_numeric(df["price"])
    df["quantity"] = pd.to_numeric(df["quantity"])

    return df
