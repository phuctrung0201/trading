"""Download OHLCV and order book data from Binance via the public REST API."""

import csv
import os
import time
from datetime import datetime, timezone

import requests

_KLINES_URL = "https://api.binance.com/api/v3/klines"
_DEPTH_URL = "https://api.binance.com/api/v3/depth"

# Map human-readable step strings to Binance interval codes.
_INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

# Seconds per step for computing how many snapshots to take.
_STEP_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}


def _instrument_to_symbol(instrument: str) -> str:
    """Convert an instrument like 'ETH-USDT-SWAP' or 'ETH-USDT' to Binance symbol 'ETHUSDT'."""
    parts = instrument.split("-")
    return (parts[0] + parts[1]).upper()


def _iso_to_ms(iso: str) -> int:
    """Convert an ISO-8601 timestamp to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def _iso_to_seconds(iso: str) -> float:
    """Convert an ISO-8601 timestamp to seconds since epoch."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.timestamp()


# ---------------------------------------------------------------------------
# OHLCV (price) download
# ---------------------------------------------------------------------------

def price(
    instrument: str,
    start: str,
    end: str,
    step: str = "1h",
    format: str = "csv",
    output_dir: str = "data",
) -> str:
    """Download OHLCV candles from Binance and save to a file.

    Parameters
    ----------
    instrument : str
        Instrument ID, e.g. ``"ETH-USDT-SWAP"`` or ``"ETH-USDT"``.
    start : str
        ISO-8601 start timestamp (inclusive).
    end : str
        ISO-8601 end timestamp (exclusive).
    step : str
        Candle interval, e.g. ``"30m"``, ``"1h"``, ``"1d"``.
    format : str
        Output format. Currently only ``"csv"`` is supported.
    output_dir : str
        Directory to write the output file into.

    Returns
    -------
    str
        Path to the saved file.
    """
    if step not in _INTERVAL_MAP:
        raise ValueError(
            f"Unsupported step '{step}'. Choose from: {', '.join(_INTERVAL_MAP)}"
        )
    if format != "csv":
        raise ValueError(f"Unsupported format '{format}'. Only 'csv' is supported.")

    symbol = _instrument_to_symbol(instrument)
    filename = f"{symbol}_{step}_{start[:10]}_{end[:10]}.csv"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"File already exists, skipping download: {filepath}")
        return filepath

    interval = _INTERVAL_MAP[step]
    start_ms = _iso_to_ms(start)
    end_ms = _iso_to_ms(end)

    all_candles: list[list] = []
    current_start = start_ms
    limit = 1000  # Binance max per request

    print(f"Downloading {instrument} ({step}) from {start} to {end} …")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms - 1,  # endTime is inclusive on Binance
            "limit": limit,
        }
        resp = requests.get(_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_candles.extend(data)
        # Move start to 1 ms after the last candle's open time to avoid overlap
        current_start = data[-1][0] + 1

        if len(data) < limit:
            break

        # Be polite to the API
        time.sleep(0.2)

    print(f"  Fetched {len(all_candles)} candles.")

    # ---- Write to CSV ----
    os.makedirs(output_dir, exist_ok=True)

    header = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for candle in all_candles:
            # Binance kline fields:
            # 0=open_time, 1=open, 2=high, 3=low, 4=close, 5=volume, …
            open_time_s = candle[0] / 1000.0
            dt = datetime.fromtimestamp(open_time_s, tz=timezone.utc)
            writer.writerow([
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                candle[1],  # open
                candle[2],  # high
                candle[3],  # low
                candle[4],  # close
                candle[5],  # volume
            ])

    print(f"  Saved to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Order book snapshots
# ---------------------------------------------------------------------------

def order_book(
    instrument: str,
    start: str,
    end: str,
    step: str = "30m",
    depth: int | str = 100,
    format: str = "csv",
    output_dir: str = "data",
) -> str:
    """Fetch order book snapshots from Binance at regular intervals.

    Because Binance only provides the *current* order book, this function
    fetches one snapshot per ``step`` between ``start`` and ``end`` using
    the kline open-times as reference timestamps.  If the time range is in
    the past the snapshots are taken immediately in sequence (the
    timestamps recorded are the kline open-times, not wall-clock time).

    Parameters
    ----------
    instrument : str
        Instrument ID, e.g. ``"ETH-USDT-SWAP"`` or ``"ETH-USDT"``.
    start : str
        ISO-8601 start timestamp (inclusive).
    end : str
        ISO-8601 end timestamp (exclusive).
    step : str
        Interval between snapshots (e.g. ``"30m"``).
    depth : int | str
        Number of bid/ask levels per snapshot (5, 10, 20, 50, 100, 500, 1000, 5000).
    format : str
        Output format. Currently only ``"csv"`` is supported.
    output_dir : str
        Directory to write the output file into.

    Returns
    -------
    str
        Path to the saved file.
    """
    depth = int(depth)
    if format != "csv":
        raise ValueError(f"Unsupported format '{format}'. Only 'csv' is supported.")
    if step not in _STEP_SECONDS:
        raise ValueError(
            f"Unsupported step '{step}'. Choose from: {', '.join(_STEP_SECONDS)}"
        )

    symbol = _instrument_to_symbol(instrument)
    filename = f"{symbol}_book_d{depth}_{step}_{start[:10]}_{end[:10]}.csv"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"File already exists, skipping download: {filepath}")
        return filepath

    start_s = _iso_to_seconds(start)
    end_s = _iso_to_seconds(end)
    step_s = _STEP_SECONDS[step]

    os.makedirs(output_dir, exist_ok=True)

    header = ["timestamp", "side", "price", "quantity"]
    rows_written = 0

    print(f"Downloading {instrument} order book (depth={depth}, step={step}) …")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        ts = start_s
        while ts < end_s:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            params = {"symbol": symbol, "limit": depth}
            resp = requests.get(_DEPTH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for bid in data.get("bids", []):
                writer.writerow([ts_str, "bid", bid[0], bid[1]])
                rows_written += 1

            for ask in data.get("asks", []):
                writer.writerow([ts_str, "ask", ask[0], ask[1]])
                rows_written += 1

            ts += step_s
            time.sleep(0.2)

    print(f"  Fetched {rows_written} rows.")
    print(f"  Saved to {filepath}")
    return filepath
