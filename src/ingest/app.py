from src.app.provider import AppProvider
from src.clickhouse.dataset import DatasetWriter, DatasetReader


class IngestApp:
    def __init__(self, provider: AppProvider):
        self._logger = provider.logger
        self._clickhouse_client = provider.clickhouse_client
        self._okx_exchange = provider.okx_exchange

        if self._clickhouse_client is None:
            raise RuntimeError("ClickHouse must be enabled for ingestion")

        self._logger.info("IngestApp ready")

    def run(self, dataset: str, start_ms: int, end_ms: int):
        reader = DatasetReader(self._clickhouse_client, dataset, self._logger)
        writer = DatasetWriter(self._clickhouse_client, dataset, self._logger)

        existing = reader.count()
        if existing > 0:
            self._logger.info(
                f"Dataset {dataset} exists with {existing} rows, refreshing"
            )
            writer.delete()

        self._logger.info(
            f"Ingesting dataset={dataset} start_ms={start_ms} end_ms={end_ms}"
        )
        total = 0
        first_ts = None
        last_ts = None
        for trade in self._okx_exchange.stream_range(start_ms, end_ms):
            writer.write(trade)
            total += 1
            if first_ts is None:
                first_ts = trade.timestamp
            last_ts = trade.timestamp
            if total % 10_000 == 0:
                self._logger.info(
                    f"Ingested {total} trades last_ts={last_ts}"
                )

        self._logger.info(
            f"Ingest complete dataset={dataset} total={total} "
            f"first_ts={first_ts} last_ts={last_ts}"
        )
