import threading
import time
from urllib.parse import quote

import requests
from src.app.logger import AppLogger


class InfluxWorker:
    def __init__(self, client, buffer_size, flush_delay, app_logger: AppLogger):
        self.client = client
        self._logger = app_logger
        self.buffer_size = int(buffer_size)
        self.flush_delay = float(flush_delay)
        self._buffer = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def queue(self, measurement, fields, timestamp, tags=None):
        item = {
            "measurement": measurement,
            "fields": fields,
            "timestamp": timestamp,
            "tags": tags or {},
        }
        should_flush = False
        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self.buffer_size:
                should_flush = True
        if should_flush:
            self.flush()

    def flush(self):
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []
        self.client._send_batch(batch)

    def close(self):
        with self._lock:
            buffered_count = len(self._buffer)
        self._logger.info(f"InfluxWorker closing buffered_count={buffered_count}")
        self._stop.set()
        self._thread.join(timeout=max(self.flush_delay, 0.1))
        self.flush()
        with self._lock:
            remaining_count = len(self._buffer)
        self._logger.info(f"InfluxWorker closed remaining_buffered_count={remaining_count}")

    def _run(self):
        while not self._stop.wait(self.flush_delay):
            self.flush()


class InfluxDBClient:
    def __init__(self, url, token, org, bucket, app_logger: AppLogger):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self._session = requests.Session()
        self._logger = app_logger
        self.worker = InfluxWorker(
            client=self,
            buffer_size=100,
            flush_delay=1.0,
            app_logger=app_logger,
        )

    def write(self, measurement, fields, timestamp, tags=None):
        self.worker.queue(measurement, fields, timestamp, tags)

    def close(self):
        self._logger.info("InfluxDBClient close requested")
        self.worker.close()
        self._session.close()
        self._logger.info("InfluxDBClient closed")

    def _send_batch(self, batch):
        lines = []
        for point in batch:
            measurement = self._escape_key(point.get("measurement", ""))
            if not measurement:
                continue

            fields = point.get("fields", {})
            if not isinstance(fields, dict) or not fields:
                continue

            field_parts = []
            for key, value in fields.items():
                encoded = self._encode_field_value(value)
                if encoded is None:
                    continue
                field_parts.append(f"{self._escape_key(key)}={encoded}")
            if not field_parts:
                continue

            tags = point.get("tags", {})
            tag_parts = []
            if isinstance(tags, dict):
                for key, value in tags.items():
                    tag_parts.append(f"{self._escape_key(key)}={self._escape_key(value)}")

            ts = point.get("timestamp")
            if isinstance(ts, (int, float)):
                timestamp_ns = int(ts)
            else:
                timestamp_ns = time.time_ns()

            if tag_parts:
                lines.append(f"{measurement},{','.join(tag_parts)} {','.join(field_parts)} {timestamp_ns}")
            else:
                lines.append(f"{measurement} {','.join(field_parts)} {timestamp_ns}")

        if not lines:
            return

        url = (
            f"{self.url}/api/v2/write"
            f"?org={quote(str(self.org))}"
            f"&bucket={quote(str(self.bucket))}"
            f"&precision=ns"
        )
        payload = "\n".join(lines)
        try:
            response = self._session.post(
                url,
                data=payload,
                headers={
                    "Authorization": f"Token {self.token}",
                    "Content-Type": "text/plain; charset=utf-8",
                },
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            # Preserve current non-throwing flush semantics.
            return

    def _escape_key(self, value):
        text = str(value)
        return (
            text.replace("\\", "\\\\")
            .replace(",", "\\,")
            .replace(" ", "\\ ")
            .replace("=", "\\=")
        )

    def _encode_field_value(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int) and not isinstance(value, bool):
            return f"{value}i"
        if isinstance(value, float):
            return repr(value)
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f"\"{text}\""
