from __future__ import annotations

from src.okx.auth import OkxAuth
from src.okx.pool import OkxClientPool


class TradeRepo:
    def __init__(self, pool: OkxClientPool, auth: OkxAuth):
        self._pool = pool
        self._auth = auth

    def recent_trades(self, inst_id: str, limit: int = 100) -> list[dict]:
        return self._pool.public_get(
            "/api/v5/market/trades",
            params={"instId": inst_id, "limit": str(limit)},
        )

    def history_trades(
        self,
        inst_id: str,
        limit: int = 100,
        after: str | None = None,
        type_param: str | None = None,
    ) -> list[dict]:
        params: dict[str, str] = {"instId": inst_id, "limit": str(limit)}
        if after is not None:
            params["after"] = after
        if type_param is not None:
            params["type"] = type_param
        return self._pool.public_get(
            "/api/v5/market/history-trades", params=params,
        )

    def fetch_candle_close(self, inst_id: str, bar: str, ts: int) -> float:
        """Falls back to history-candles when recent-candles has no data."""
        params = {"instId": inst_id, "bar": bar, "after": str(ts + 1), "limit": "1"}
        try:
            candles = self._pool.public_get("/api/v5/market/candles", params=params)
            if candles:
                return float(candles[0][4])
            candles = self._pool.public_get("/api/v5/market/history-candles", params=params)
            if candles:
                return float(candles[0][4])
        except Exception:
            pass
        return 0.0

    def fetch_range(self, inst_id: str, start_ms: int, end_ms: int) -> list[dict]:
        raw = self._page_backwards(inst_id, start_ms, end_ms)
        return self._dedup_and_sort(raw, start_ms)

    def fetch_ranges(
        self,
        specs: list[tuple[str, int, int]],
        max_workers: int | None = None,
    ) -> list[list[dict]]:
        if not specs:
            return []
        return self._pool.map(self._fetch_range_item, specs, max_workers=max_workers)

    def stream_range(self, inst_id: str, start_ms: int, end_ms: int) -> list[dict]:
        one_hour_ms = 3_600_000
        specs: list[tuple[str, int, int]] = []
        cursor = start_ms
        while cursor < end_ms:
            r_end = min(cursor + one_hour_ms, end_ms)
            specs.append((inst_id, cursor, r_end))
            cursor = r_end

        chunk_results = self.fetch_ranges(specs)
        merged: list[dict] = []
        for chunk in chunk_results:
            merged.extend(chunk)
        return self._dedup_and_sort(merged, start_ms)

    def _page_backwards(
        self, inst_id: str, start_ms: int, end_ms: int,
    ) -> list[dict]:
        trades: list[dict] = []
        cursor = str(end_ms)
        while True:
            data = self._pool.public_get(
                "/api/v5/market/history-trades",
                params={
                    "instId": inst_id,
                    "limit": "100",
                    "after": cursor,
                    "type": "2",
                },
            )
            if not data:
                break
            trades.extend(data)
            oldest_ts = int(data[-1]["ts"])
            if oldest_ts <= start_ms or len(data) < 100:
                break
            if str(oldest_ts) == cursor:
                break
            cursor = str(oldest_ts)
        return trades

    def _fetch_range_item(
        self, pool: OkxClientPool, spec: tuple[str, int, int],
    ) -> list[dict]:
        inst_id, start_ms, end_ms = spec
        trades: list[dict] = []
        cursor = str(end_ms)
        while True:
            data = pool.public_get(
                "/api/v5/market/history-trades",
                params={
                    "instId": inst_id,
                    "limit": "100",
                    "after": cursor,
                    "type": "2",
                },
            )
            if not data:
                break
            trades.extend(data)
            oldest_ts = int(data[-1]["ts"])
            if oldest_ts <= start_ms or len(data) < 100:
                break
            if str(oldest_ts) == cursor:
                break
            cursor = str(oldest_ts)
        return self._dedup_and_sort(trades, start_ms)

    @staticmethod
    def _dedup_and_sort(trades: list[dict], start_ms: int) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for t in trades:
            tid = t["tradeId"]
            if tid not in seen:
                seen.add(tid)
                if int(t["ts"]) >= start_ms:
                    unique.append(t)
        unique.sort(key=lambda t: int(t["ts"]))
        return unique

    def place_order(
        self,
        instrument: str,
        side: str,
        size: float,
        order_type: str = "market",
    ) -> list:
        body = {
            "instId": instrument,
            "tdMode": "cross",
            "side": side,
            "ordType": order_type,
            "sz": str(size),
        }
        return self._auth.signed_request("POST", "/api/v5/trade/order", body)

    def get_order(self, instrument: str, order_id: str) -> dict:
        path = f"/api/v5/trade/order?instId={instrument}&ordId={order_id}"
        data = self._auth.signed_request("GET", path)
        if not data:
            return {}
        return data[0]

    def close_position(self, instrument: str):
        body = {"instId": instrument, "mgnMode": "cross"}
        return self._auth.signed_request(
            "POST", "/api/v5/trade/close-position", body,
        )
