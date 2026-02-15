import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
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

    def _signed_request(self, method, path, body=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        body_str = json.dumps(body) if body is not None else ""
        prehash = timestamp + method.upper() + path + body_str
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode("utf-8"),
                prehash.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if self.demo:
            headers["x-simulated-trading"] = "1"

        if method.upper() == "GET":
            resp = self._session.get(
                f"{_BASE_URL}{path}",
                headers=headers,
                timeout=15,
            )
        else:
            resp = self._session.post(
                f"{_BASE_URL}{path}",
                headers=headers,
                data=body_str,
                timeout=15,
            )

        resp.raise_for_status()
        result = resp.json()
        if str(result.get("code", "")) != "0":
            raise RuntimeError(
                f"OKX request failed: {result.get('code')} {result.get('msg')}"
            )
        return result.get("data", [])

    def place_order(self, instrument, side, size, order_type="market"):
        body = {
            "instId": instrument,
            "tdMode": "cross",
            "side": side,
            "ordType": order_type,
            "sz": str(size),
        }
        return self._signed_request("POST", "/api/v5/trade/order", body)

    def close_position(self, instrument, side):
        body = {
            "instId": instrument,
            "mgnMode": "cross",
            "posSide": side,
        }
        return self._signed_request("POST", "/api/v5/trade/close-position", body)

    def get_asset(self, currency="USDT"):
        data = self._signed_request("GET", f"/api/v5/account/balance?ccy={currency}")
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for detail in details:
            if detail.get("ccy") == currency:
                return float(detail.get("availBal", 0))
        return 0.0

    def set_leverage(self, instrument, leverage, mgn_mode="cross"):
        body = {
            "instId": instrument,
            "lever": str(leverage),
            "mgnMode": mgn_mode,
        }
        return self._signed_request("POST", "/api/v5/account/set-leverage", body)

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

    def stream_history(self, instrument, start, end, step):
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
        for c in all_candles:
            ts = int(c[0])
            confirmed = len(c) > 8 and str(c[8]) == "1"
            if start_ms <= ts <= end_ms and confirmed:
                yield OHCLV(
                    timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )

    def stream_prices(self, instrument, step):
        last_ts = None
        while True:
            raw = self.candles(instrument=instrument, bar=step, limit=2)
            for candle in raw:
                ts = int(candle[0])
                confirmed = len(candle) > 8 and str(candle[8]) == "1"
                if confirmed and ts != last_ts:
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
