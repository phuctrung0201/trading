import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests
from requests import HTTPError

from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position, OpenResult
from src.exchange.position import PositionTracker
from src.okx.auth import OkxAuth, _BASE_URL


class _RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, rps: float):
        self._min_interval = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = self._min_interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class OkxExchange(ExchangeAdapter):
    """OKX exchange adapter: market data, order execution, and position tracking."""

    MAX_RETRIES = 5
    MARGIN_REDUCE = 0.8

    def __init__(self, okx_client, logger=None):
        self._auth: OkxAuth = okx_client.auth
        self._trading = okx_client.trading
        self._account = okx_client.account
        self._logger = logger
        self._instrument: str | None = None
        self._leverage: int = 1
        self._leverage_set = False
        self._ct_val: float | None = None
        self._lot_sz: float | None = None
        self._tracker: PositionTracker | None = None

    def bootstrap(self, instrument: str, leverage: int = 1):
        self._instrument = instrument
        self._leverage = leverage
        initial_equity = float(self._account.get_asset(currency="USDT"))
        self._tracker = PositionTracker(initial_equity)

    # -- market data --------------------------------------------------------

    def _request_trades(self, endpoint: str, limit: int = 100,
                        after=None, before=None, type_param=None) -> list:
        params = {"instId": self._instrument, "limit": str(limit)}
        if after is not None:
            params["after"] = str(after)
        if before is not None:
            params["before"] = str(before)
        if type_param is not None:
            params["type"] = str(type_param)

        attempts = 0
        while True:
            try:
                resp = self._auth.session.get(
                    f"{_BASE_URL}{endpoint}", params=params, timeout=15,
                )
                if resp.status_code != 429:
                    resp.raise_for_status()
                    body = resp.json()
                    if str(body.get("code", "")) != "0":
                        raise RuntimeError(
                            f"OKX trades request failed: {body.get('code')} {body.get('msg')}"
                        )
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
            except (requests.exceptions.RequestException, HTTPError) as exc:
                attempts += 1
                if attempts >= 6:
                    raise
                delay = min(30.0, 1.0 * (2 ** (attempts - 1)))
                time.sleep(delay)

    def _recent_trades(self, limit: int = 100) -> list:
        return self._request_trades(
            endpoint="/api/v5/market/trades", limit=limit,
        )

    def _history_trades(self, limit: int = 100,
                        after=None, before=None, type_param=None) -> list:
        return self._request_trades(
            endpoint="/api/v5/market/history-trades",
            limit=limit, after=after, before=before, type_param=type_param,
        )

    @staticmethod
    def _normalize_trade(raw: dict) -> MarketTrade:
        return MarketTrade(
            trade_id=raw["tradeId"],
            timestamp=raw["ts"],
            price=float(raw["px"]),
            size=float(raw["sz"]),
            side=raw["side"],
        )

    def _get_mark_price(self, inst_type: str = "SWAP") -> float:
        data = self._auth.public_get(
            "/api/v5/public/mark-price",
            params={"instType": inst_type, "instId": self._instrument},
        )
        if not data:
            raise RuntimeError(f"Mark price not found: {self._instrument}")
        return float(data[0]["markPx"])

    def _get_instrument(self, inst_type: str = "SWAP") -> dict:
        data = self._auth.public_get(
            "/api/v5/public/instruments",
            params={"instType": inst_type, "instId": self._instrument},
        )
        if not data:
            raise RuntimeError(f"Instrument not found: {self._instrument}")
        return data[0]

    _HISTORY_WORKERS = 4
    _HISTORY_MAX_RPS = 8

    def _log(self, msg: str):
        if self._logger is not None:
            self._logger.info(msg)

    def _fetch_range(self, range_start_ms: int, range_end_ms: int,
                     limiter: "_RateLimiter") -> list[dict]:
        """Fetch all history trades within a time range, paging backwards."""
        trades: list[dict] = []
        cursor = str(range_end_ms)
        batch = 0
        while True:
            limiter.wait()
            raw = self._history_trades(limit=100, after=cursor, type_param=2)
            if not raw:
                break
            batch += 1
            trades.extend(raw)
            oldest_ts = int(raw[-1]["ts"])
            self._log(
                f"fetch_range batch={batch} got={len(raw)} "
                f"range=[{range_start_ms}..{range_end_ms}] oldest_ts={oldest_ts} total={len(trades)}"
            )
            if oldest_ts <= range_start_ms or len(raw) < 100:
                break
            if str(oldest_ts) == cursor:
                break
            cursor = str(oldest_ts)
        return trades

    def _stream_range(self, start_ms: int, end_ms: int):
        total_ms = end_ms - start_ms
        n = min(self._HISTORY_WORKERS, max(1, total_ms // 60_000))
        chunk_ms = total_ms // n

        ranges = []
        for i in range(n):
            r_start = start_ms + i * chunk_ms
            r_end = end_ms if i == n - 1 else start_ms + (i + 1) * chunk_ms
            ranges.append((r_start, r_end))

        self._log(f"stream_range workers={n} chunks={len(ranges)}")
        limiter = _RateLimiter(self._HISTORY_MAX_RPS)
        all_trades: list[dict] = []

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(self._fetch_range, r_start, r_end, limiter): i
                for i, (r_start, r_end) in enumerate(ranges)
            }
            for future in as_completed(futures):
                chunk_trades = future.result()
                chunk_idx = futures[future]
                self._log(f"stream_range chunk={chunk_idx} fetched={len(chunk_trades)}")
                all_trades.extend(chunk_trades)

        seen: set[str] = set()
        unique: list[dict] = []
        for t in all_trades:
            tid = t["tradeId"]
            if tid not in seen:
                seen.add(tid)
                unique.append(t)

        self._log(f"stream_range raw={len(all_trades)} unique={len(unique)}")
        unique.sort(key=lambda t: int(t["ts"]))
        for t in unique:
            if int(t["ts"]) >= start_ms:
                yield self._normalize_trade(t)

    def stream_history(self, depth_sec: int):
        start_ms = int(
            (datetime.now(timezone.utc) - timedelta(seconds=depth_sec)).timestamp() * 1000
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        yield from self._stream_range(start_ms, now_ms)

    def stream_range(self, start_ms: int, end_ms: int):
        yield from self._stream_range(start_ms, end_ms)

    def stream_prices(self):
        last_trade_id: str | None = None
        while True:
            try:
                raw = self._recent_trades(limit=100)
                raw.sort(key=lambda t: int(t["ts"]))
                for t in raw:
                    tid = t["tradeId"]
                    if last_trade_id is not None and int(tid) <= int(last_trade_id):
                        continue
                    last_trade_id = tid
                    yield self._normalize_trade(t)
            except Exception:
                time.sleep(1)
            time.sleep(0.2)

    # -- contract helpers ---------------------------------------------------

    def _ensure_contract_info(self):
        if self._ct_val is None:
            info = self._get_instrument()
            self._ct_val = float(info["ctVal"])
            self._lot_sz = float(info["lotSz"])

    def _usd_to_contracts(self, usd_size: float) -> float:
        self._ensure_contract_info()
        assert self._ct_val is not None and self._lot_sz is not None
        mark_price = self._get_mark_price()
        contracts = usd_size / (self._ct_val * mark_price)
        lots = int(contracts / self._lot_sz) * self._lot_sz
        return round(lots, 8)

    @staticmethod
    def _is_margin_error(message: str) -> bool:
        return "Insufficient" in message and "margin" in message

    # -- exchange interface -------------------------------------------------

    def set_price(self, price: float):
        self._tracker.set_price(price)

    def fetch_asset(self, asset: str) -> float:
        return self._account.get_asset(currency=asset)

    def fetch_position(self) -> Position | None:
        data = self._account.get_positions(self._instrument)
        if not data:
            return None
        pos_data = data[0]
        pos_qty = float(pos_data.get("pos", "0"))
        if pos_qty == 0:
            return None
        self._ensure_contract_info()
        assert self._ct_val is not None
        avg_px = float(pos_data.get("avgPx", "0"))
        notional = abs(pos_qty) * self._ct_val * avg_px
        side = "buy" if pos_qty > 0 else "sell"
        return Position(side=side, size=notional, price=avg_px if avg_px > 0 else None)

    def open(self, position: Position) -> OpenResult:
        if not self._leverage_set:
            try:
                self._account.set_leverage(self._instrument, self._leverage)
                self._leverage_set = True
            except Exception as exc:
                return OpenResult(success=False, message=f"leverage set failed: {exc}")

        side = position.side
        usd_size = position.size
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                contracts = self._usd_to_contracts(usd_size)
                if contracts <= 0:
                    return OpenResult(
                        success=False,
                        message=f"size too small usd={usd_size:.4f} contracts={contracts}",
                    )
                data = self._trading.place_order(
                    instrument=self._instrument,
                    side=side,
                    size=contracts,
                )
                order_id = None
                if data and isinstance(data, list) and len(data) > 0:
                    order_id = data[0].get("ordId")

                fill_price = None
                if order_id:
                    order_detail = self._trading.get_order(self._instrument, order_id)
                    avg_px = order_detail.get("avgPx")
                    if avg_px and avg_px != "":
                        fill_price = float(avg_px)

                filled = Position(side=side, size=usd_size, price=fill_price)
                self._tracker.open(filled)
                return OpenResult(success=True, position=filled)
            except Exception as exc:
                last_error = str(exc)
                if self._is_margin_error(last_error):
                    usd_size *= self.MARGIN_REDUCE
                delay = min(30.0, 1.0 * (2 ** attempt))
                time.sleep(delay)

        return OpenResult(success=False, message=last_error)

    def close(self, position: Position) -> bool:
        for attempt in range(self.MAX_RETRIES):
            try:
                self._trading.close_position(instrument=self._instrument)
                self._tracker.close()
                return True
            except Exception:
                delay = min(30.0, 1.0 * (2 ** attempt))
                time.sleep(delay)
        return False

    def get_asset(self, asset: str) -> float:
        return self._tracker.get_asset(asset)

    def get_position(self) -> Position | None:
        return self._tracker.get_position()

    def get_equity(self) -> float:
        return self._tracker.get_equity()

    def unrealized_pnl(self) -> float:
        return self._tracker.unrealized_pnl()
