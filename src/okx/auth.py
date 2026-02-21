import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import time

import requests

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


class OkxAuth:
    def __init__(self, api_key: str, secret_key: str, passphrase: str, demo: bool):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.demo = demo
        self._session = requests.Session()
        self._api_callback = None

    def set_api_callback(self, callback):
        self._api_callback = callback

    def _notify_api(self, latency_ms: int, response_code: int, source: str):
        if self._api_callback:
            self._api_callback(latency_ms=latency_ms, response_code=response_code, source=source)

    def signed_request(self, method: str, path: str, body=None):
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

        attempts = 0
        while True:
            try:
                t0 = time.monotonic()
                if method.upper() == "GET":
                    resp = self._session.get(f"{_BASE_URL}{path}", headers=headers, timeout=15)
                else:
                    resp = self._session.post(f"{_BASE_URL}{path}", headers=headers, data=body_str, timeout=15)
                latency_ms = int((time.monotonic() - t0) * 1000)
                self._notify_api(latency_ms=latency_ms, response_code=resp.status_code, source=path)

                resp.raise_for_status()
                result = resp.json()
                if str(result.get("code", "")) != "0":
                    detail = result.get("msg", "")
                    data = result.get("data")
                    if isinstance(data, list) and data:
                        detail = data[0].get("sMsg", detail)
                    raise RuntimeError(f"OKX request failed: {result.get('code')} {detail}")
                return result.get("data", [])
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as exc:
                attempts += 1
                if attempts >= 5:
                    raise
                delay = min(30.0, 1.0 * (2 ** (attempts - 1)))
                time.sleep(delay)

    def public_get(self, path: str, params: dict | None = None):
        attempts = 0
        while True:
            try:
                resp = self._session.get(f"{_BASE_URL}{path}", params=params, timeout=15)
                resp.raise_for_status()
                return resp.json().get("data", [])
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as exc:
                attempts += 1
                if attempts >= 5:
                    raise
                delay = min(30.0, 1.0 * (2 ** (attempts - 1)))
                time.sleep(delay)

    @property
    def session(self) -> requests.Session:
        return self._session
