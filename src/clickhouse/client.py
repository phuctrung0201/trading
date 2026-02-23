import io
import queue
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.app.logger import AppLogger
from src.clickhouse.schema import TABLE_SCHEMAS


def _rows_to_parquet_bytes(table_name: str, rows: list[dict]) -> bytes:
    schema = TABLE_SCHEMAS[table_name]
    columns: dict[str, list] = {field.name: [] for field in schema}
    for row in rows:
        for field in schema:
            val = row.get(field.name)
            if pa.types.is_timestamp(field.type) and isinstance(val, (int, float)):
                val = int(val)
            columns[field.name].append(val)

    arrays = []
    for field in schema:
        arrays.append(pa.array(columns[field.name], type=field.type))
    arrow_table = pa.table(arrays, schema=schema)
    buf = io.BytesIO()
    pq.write_table(arrow_table, buf, compression="snappy")
    return buf.getvalue()


_SENTINEL = None
_BUFFER_SIZE = 200
_FLUSH_DELAY = 0.5


class WriteBuffer:
    def __init__(self, client, buffer_size: int, flush_delay: float,
                 app_logger: AppLogger):
        self._client = client
        self._logger = app_logger
        self._buffer_size = buffer_size
        self._flush_delay = flush_delay
        self._inbox: queue.Queue[tuple[str, dict] | None] = queue.Queue()
        self._buffers: dict[str, list[dict]] = {}
        self._buffer_ages: dict[str, float] = {}
        self._count = 0
        self._count_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def queue(self, table: str, row: dict):
        self._inbox.put((table, row))
        with self._count_lock:
            self._count += 1

    def buffer_size_total(self) -> int:
        with self._count_lock:
            return self._count

    def close(self):
        self._logger.info(f"WriteBuffer closing buffered_count={self.buffer_size_total()}")
        self._stop.set()
        self._inbox.put(_SENTINEL)
        self._thread.join(timeout=30)
        self._logger.info(f"WriteBuffer closed remaining_count={self.buffer_size_total()}")

    def _drain_inbox(self):
        while True:
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                continue
            assert item is not None
            table, row = item
            buf = self._buffers.setdefault(table, [])
            if not buf:
                self._buffer_ages[table] = time.monotonic()
            buf.append(row)

    def _flush_buffer(self, table: str):
        rows = self._buffers.pop(table, [])
        self._buffer_ages.pop(table, None)
        if not rows:
            return
        try:
            data = _rows_to_parquet_bytes(table, rows)
            self._client._send_parquet(table, data)
        except Exception as exc:
            self._logger.error(f"WriteBuffer flush failed table={table} rows={len(rows)}: {exc}")
        with self._count_lock:
            self._count -= len(rows)

    def _flush_all(self):
        for table in list(self._buffers):
            self._flush_buffer(table)

    def _run(self):
        while True:
            try:
                item = self._inbox.get(timeout=self._flush_delay)
                if item is _SENTINEL:
                    if self._stop.is_set():
                        self._drain_inbox()
                        self._flush_all()
                        return
                    continue
                assert item is not None
                table, row = item
                buf = self._buffers.setdefault(table, [])
                if not buf:
                    self._buffer_ages[table] = time.monotonic()
                buf.append(row)
            except queue.Empty:
                pass

            self._drain_inbox()

            now = time.monotonic()
            for table in list(self._buffers):
                buf = self._buffers.get(table, [])
                age = now - self._buffer_ages.get(table, now)
                if len(buf) >= self._buffer_size or age >= self._flush_delay:
                    self._flush_buffer(table)


class ClickHouseClient:
    def __init__(self, url: str, database: str, user: str, password: str,
                 app_logger: AppLogger):
        self.url = url
        self.database = database
        self.user = user
        self.password = password
        self._session = requests.Session()
        self._logger = app_logger
        self.worker = WriteBuffer(
            client=self,
            buffer_size=_BUFFER_SIZE,
            flush_delay=_FLUSH_DELAY,
            app_logger=app_logger,
        )

    def write(self, table: str, row: dict):
        self.worker.queue(table, row)

    def buffer_size_total(self) -> int:
        return self.worker.buffer_size_total()

    def close(self):
        self._logger.info("ClickHouseClient close requested")
        self.worker.close()
        self._session.close()
        self._logger.info("ClickHouseClient closed")

    def _send_parquet(self, table: str, data: bytes):
        try:
            resp = self._session.post(
                self.url,
                params={
                    "database": self.database,
                    "user": self.user,
                    "password": self.password,
                    "query": f"INSERT INTO {self.database}.{table} FORMAT Parquet",
                },
                data=data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=120,
            )
            resp.raise_for_status()
            self._logger.info(f"Parquet insert table={table} bytes={len(data)}")
        except Exception as exc:
            self._logger.error(f"Parquet insert failed table={table}: {exc}")

    def _exec_strict(self, query: str):
        """Execute query, raise on failure."""
        resp = self._session.post(
            self.url,
            params={"database": self.database, "user": self.user, "password": self.password},
            data=query,
            timeout=10,
        )
        resp.raise_for_status()
