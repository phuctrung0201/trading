from collections.abc import Iterator

from src.app.logger import AppLogger
from src.clickhouse.client import ClickHouseClient
from src.exchange.dto import MarketTrade


class DatasetReader:
    """Stream MarketTrade rows from the market_trade table by dataset name."""

    _PAGE_SIZE = 10_000

    def __init__(self, client: ClickHouseClient, dataset: str, logger: AppLogger):
        self._client = client
        self._dataset = dataset
        self._logger = logger

    def stream(self) -> Iterator[MarketTrade]:
        offset = 0
        total = 0
        while True:
            query = (
                f"SELECT trade_id, timestamp, price, size, side "
                f"FROM {self._client.database}.market_trade "
                f"WHERE dataset = '{self._dataset}' "
                f"ORDER BY timestamp "
                f"LIMIT {self._PAGE_SIZE} OFFSET {offset} "
                f"FORMAT TabSeparatedWithNames"
            )
            resp = self._client._session.post(
                self._client.url,
                params={
                    "database": self._client.database,
                    "user": self._client.user,
                    "password": self._client.password,
                },
                data=query,
                timeout=60,
            )
            resp.raise_for_status()
            lines = resp.text.strip().splitlines()
            if len(lines) <= 1:
                break
            rows = lines[1:]
            for line in rows:
                trade_id, ts, price, size, side = line.split("\t")
                total += 1
                yield MarketTrade(
                    trade_id=trade_id,
                    timestamp=ts,
                    price=float(price),
                    size=float(size),
                    side=side,
                )
            if len(rows) < self._PAGE_SIZE:
                break
            offset += self._PAGE_SIZE
        self._logger.info(f"DatasetReader done dataset={self._dataset} total={total}")

    def count(self) -> int:
        query = (
            f"SELECT count() FROM {self._client.database}.market_trade "
            f"WHERE dataset = '{self._dataset}'"
        )
        resp = self._client._session.post(
            self._client.url,
            params={
                "database": self._client.database,
                "user": self._client.user,
                "password": self._client.password,
            },
            data=query,
            timeout=10,
        )
        resp.raise_for_status()
        return int(resp.text.strip())


class DatasetWriter:
    """Write MarketTrade rows into the market_trade table tagged with a dataset name."""

    def __init__(self, client: ClickHouseClient, dataset: str, logger: AppLogger):
        self._client = client
        self._dataset = dataset
        self._logger = logger

    def delete(self):
        query = (
            f"ALTER TABLE {self._client.database}.market_trade "
            f"DELETE WHERE dataset = '{self._dataset}'"
        )
        self._client._exec_strict(query)
        self._logger.info(f"DatasetWriter deleted dataset={self._dataset}")

    def write(self, trade: MarketTrade):
        row = {
            "dataset": self._dataset,
            "trade_id": trade.trade_id,
            "timestamp": int(trade.timestamp),
            "price": trade.price,
            "size": trade.size,
            "side": trade.side,
        }
        self._client.write("market_trade", row)
