import csv
import os

from src.exchange.dto import MarketTrade


DATA_DIR = "data"
CANDLE_FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


def data_path(instrument: str, step: str, start: str, end: str) -> str:
    safe = lambda s: str(s).replace("/", "_").replace(":", "-").replace(" ", "_")
    filename = f"{safe(instrument)}_{step}_{safe(start)}_{safe(end)}.csv"
    return os.path.join(DATA_DIR, filename)


def download_history(exchange, start: str, end: str, step: str,
                     path: str, logger):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Downloading history to {path}")
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDLE_FIELDS)
        writer.writeheader()
        for trade in exchange.stream_history(start=start, end=end, step=step):
            writer.writerow({
                "timestamp": trade.timestamp,
                "open": trade.open,
                "high": trade.high,
                "low": trade.low,
                "close": trade.close,
                "volume": trade.volume,
            })
            count += 1
    logger.info(f"Downloaded {count} candles to {path}")


def load_candles(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield MarketTrade(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
