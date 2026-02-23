from abc import ABC, abstractmethod

from src.clickhouse.measurement import OpsMeasurement


class Recorder(ABC):
    @abstractmethod
    def record(self, measurable):
        raise NotImplementedError


class ClickHouseRecorder(Recorder):
    def __init__(self, clickhouse_client, session_id: str):
        self.clickhouse_client = clickhouse_client
        self.session_id = session_id
        self.setup_name: str = ""
        self.instrument: str = ""
        self.strategy: str = ""
        self.indicator: str = "ema"

    def bootstrap(self, setup_name: str, instrument: str, strategy: str,
                  indicator: str = "ema"):
        self.setup_name = setup_name
        self.instrument = instrument
        self.strategy = strategy
        self.indicator = indicator

    def _table_for(self, measurable) -> str:
        if isinstance(measurable, OpsMeasurement):
            return "ops"
        return "trade_event"

    def record(self, measurable):
        fields = measurable.to_dict()
        timestamp = getattr(measurable, "timestamp", None)
        row = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "setup": self.setup_name,
            "instrument": self.instrument,
            "strategy": self.strategy,
            "indicator": self.indicator,
            **fields,
        }
        table = self._table_for(measurable)
        if self.clickhouse_client is not None:
            self.clickhouse_client.write(table, row)
        return row
