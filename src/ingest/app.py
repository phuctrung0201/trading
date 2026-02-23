from src.app.logger import AppLogger
from src.clickhouse.client import ClickHouseClient
from src.clickhouse.dataset import DatasetWriter, DatasetReader
from src.ingest.poller import IngestPoller


class IngestApp:
    def __init__(
        self,
        poller: IngestPoller,
        clickhouse_client: ClickHouseClient,
        logger: AppLogger,
    ):
        self._poller = poller
        self._clickhouse_client = clickhouse_client
        self._logger = logger

    def run(self, dataset: str, start_ms: int, end_ms: int):
        reader = DatasetReader(self._clickhouse_client, dataset, self._logger)
        writer = DatasetWriter(self._clickhouse_client, dataset, self._logger)

        existing = reader.count()
        if existing > 0:
            self._logger.info(f"Dataset {dataset} has {existing} rows, refreshing")
            writer.delete()

        trades = self._poller.poll(start_ms, end_ms)
        for trade in trades:
            writer.write(trade)

        self._logger.info(f"Ingest complete dataset={dataset}")
