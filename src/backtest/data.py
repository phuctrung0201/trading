import csv
import os

from src.exchange.dto import MarketTrade


DATA_DIR = "data"
TRADE_FIELDS = ["trade_id", "timestamp", "price", "size", "side"]


def data_path(instrument: str, depth_ts: int) -> str:
    safe = lambda s: str(s).replace("/", "_").replace(":", "-").replace(" ", "_")
    filename = f"{safe(instrument)}_depth_{depth_ts}m.csv"
    return os.path.join(DATA_DIR, filename)


def download_history(exchange, depth_ts: int, path: str, logger):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logger.info(f"Downloading trade history depth_ts={depth_ts}m to {path}")
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        for trade in exchange.stream_history(depth_ts=depth_ts):
            writer.writerow({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "price": trade.price,
                "size": trade.size,
                "side": trade.side,
            })
            count += 1
    logger.info(f"Downloaded {count} trades to {path}")


def load_trades(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield MarketTrade(
                trade_id=row["trade_id"],
                timestamp=row["timestamp"],
                price=float(row["price"]),
                size=float(row["size"]),
                side=row["side"],
            )
