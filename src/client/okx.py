from datetime import datetime, timezone
import time

import requests
from requests import HTTPError

from .ohclv import OHCLV


_BASE_URL = "https://www.okx.com"


def _to_epoch_ms(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class OkxClient:
    def __init__(self, api_key, secret_key, passphrase, demo):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.demo = demo
        self._session = requests.Session()

    def candles(self, instrument, bar="1m", limit=100, after=None, before=None):
        return self._request_candles(
            endpoint="/api/v5/market/candles",
            instrument=instrument,
            bar=bar,
            limit=limit,
            after=after,
            before=before,
        )

    def history_candles(self, instrument, bar="1m", limit=100, after=None, before=None):
        return self._request_candles(
            endpoint="/api/v5/market/history-candles",
            instrument=instrument,
            bar=bar,
            limit=limit,
            after=after,
            before=before,
        )

    def _request_candles(self, endpoint, instrument, bar="1m", limit=100, after=None, before=None):
        params = {
            "instId": instrument,
            "bar": bar,
            "limit": str(limit),
        }
        if after is not None:
            params["after"] = str(after)
        if before is not None:
            params["before"] = str(before)

        attempts = 0
        while True:
            resp = self._session.get(
                f"{_BASE_URL}{endpoint}",
                params=params,
                timeout=15,
            )
            if resp.status_code != 429:
                resp.raise_for_status()
                body = resp.json()
                if str(body.get("code", "")) != "0":
                    raise RuntimeError(f"OKX candles request failed: {body.get('code')} {body.get('msg')}")
                return body.get("data", [])

            attempts += 1
            if attempts >= 6:
                raise HTTPError(
                    f"429 Client Error: Too Many Requests for url: {resp.url}",
                    response=resp,
                )

            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None and retry_after.isdigit():
                delay = max(1.0, float(retry_after))
            else:
                delay = min(30.0, 1.5 * (2 ** (attempts - 1)))
            time.sleep(delay)

    def get_prices(self, instrument, start, end, step):
        start_ms = _to_epoch_ms(start)
        end_ms = _to_epoch_ms(end)
        if end_ms < start_ms:
            raise ValueError("backtest.end must be greater than backtest.start")

        all_candles = []
        cursor = str(end_ms)
        while True:
            raw = self.history_candles(
                instrument=instrument,
                bar=step,
                limit=100,
                before=str(start_ms - 1),
                after=cursor,
            )
            if not raw:
                break

            all_candles.extend(raw)
            oldest_ts = int(raw[-1][0])
            if oldest_ts <= start_ms or len(raw) < 100:
                break
            if str(oldest_ts) == cursor:
                break
            cursor = str(oldest_ts)
            time.sleep(0.2)

        all_candles.sort(key=lambda c: int(c[0]))
        windowed = [c for c in all_candles if start_ms <= int(c[0]) <= end_ms]
        return [
            OHCLV(
                timestamp=datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]),
            )
            for c in windowed
        ]

    def stream_prices(self, instrument, step):
        last_ts = None
        while True:
            raw = self.candles(instrument=instrument, bar=step, limit=1)
            if raw:
                candle = raw[0]
                ts = int(candle[0])
                if ts != last_ts:
                    last_ts = ts
                    yield OHCLV(
                        timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5]),
                    )
            time.sleep(1.0)
