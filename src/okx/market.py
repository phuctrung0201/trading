import time
from datetime import datetime, timezone

from requests import HTTPError

from src.exchange.types import MarketTrade
from src.okx.auth import OkxAuth, _BASE_URL, _to_epoch_ms


class OkxMarket:
    def __init__(self, auth: OkxAuth):
        self._auth = auth

    def candles(self, instrument: str, bar: str = "1m", limit: int = 100,
                after=None, before=None) -> list:
        return self._request_candles(
            endpoint="/api/v5/market/candles",
            instrument=instrument, bar=bar, limit=limit,
            after=after, before=before,
        )

    def history_candles(self, instrument: str, bar: str = "1m", limit: int = 100,
                        after=None, before=None) -> list:
        return self._request_candles(
            endpoint="/api/v5/market/history-candles",
            instrument=instrument, bar=bar, limit=limit,
            after=after, before=before,
        )

    def get_mark_price(self, instrument: str, inst_type: str = "SWAP") -> float:
        data = self._auth.public_get(
            "/api/v5/public/mark-price",
            params={"instType": inst_type, "instId": instrument},
        )
        if not data:
            raise RuntimeError(f"Mark price not found: {instrument}")
        return float(data[0]["markPx"])

    def get_instrument(self, instrument: str, inst_type: str = "SWAP") -> dict:
        data = self._auth.public_get(
            "/api/v5/public/instruments",
            params={"instType": inst_type, "instId": instrument},
        )
        if not data:
            raise RuntimeError(f"Instrument not found: {instrument}")
        return data[0]

    def stream_history(self, instrument: str, start, end, step: str):
        start_ms = _to_epoch_ms(start)
        end_ms = _to_epoch_ms(end)
        if end_ms < start_ms:
            raise ValueError("backtest.end must be greater than backtest.start")

        all_candles = []
        cursor = str(end_ms)
        while True:
            raw = self.history_candles(
                instrument=instrument, bar=step, limit=100,
                before=str(start_ms - 1), after=cursor,
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
                yield MarketTrade(
                    timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )

    def stream_prices(self, instrument: str, step: str):
        last_ts = None
        while True:
            raw = self.candles(instrument=instrument, bar=step, limit=2)
            for candle in raw:
                ts = int(candle[0])
                confirmed = len(candle) > 8 and str(candle[8]) == "1"
                if confirmed and ts != last_ts:
                    last_ts = ts
                    yield MarketTrade(
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

    def _request_candles(self, endpoint: str, instrument: str, bar: str = "1m",
                         limit: int = 100, after=None, before=None) -> list:
        params = {"instId": instrument, "bar": bar, "limit": str(limit)}
        if after is not None:
            params["after"] = str(after)
        if before is not None:
            params["before"] = str(before)

        attempts = 0
        while True:
            resp = self._auth.session.get(f"{_BASE_URL}{endpoint}", params=params, timeout=15)
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
