import time
from datetime import datetime, timezone

from requests import HTTPError

from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position, OpenResult
from src.exchange.position import PositionTracker
from src.okx.auth import OkxAuth, _BASE_URL, _to_epoch_ms


class OkxExchange(ExchangeAdapter):
    """OKX exchange adapter: market data, order execution, and position tracking."""

    MAX_RETRIES = 3
    MARGIN_REDUCE = 0.8

    def __init__(self, okx_client, instrument: str, leverage: int = 1):
        initial_equity = float(okx_client.account.get_asset(currency="USDT"))
        self._tracker = PositionTracker(initial_equity)
        self._auth: OkxAuth = okx_client.auth
        self._trading = okx_client.trading
        self._account = okx_client.account
        self._instrument = instrument
        self._leverage = leverage
        self._leverage_set = False
        self._ct_val: float | None = None
        self._lot_sz: float | None = None

    # -- market data --------------------------------------------------------

    def _request_candles(self, endpoint: str, bar: str = "1m",
                         limit: int = 100, after=None, before=None) -> list:
        params = {"instId": self._instrument, "bar": bar, "limit": str(limit)}
        if after is not None:
            params["after"] = str(after)
        if before is not None:
            params["before"] = str(before)

        attempts = 0
        while True:
            resp = self._auth.session.get(
                f"{_BASE_URL}{endpoint}", params=params, timeout=15,
            )
            if resp.status_code != 429:
                resp.raise_for_status()
                body = resp.json()
                if str(body.get("code", "")) != "0":
                    raise RuntimeError(
                        f"OKX candles request failed: {body.get('code')} {body.get('msg')}"
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

    def _candles(self, bar: str = "1m", limit: int = 100,
                 after=None, before=None) -> list:
        return self._request_candles(
            endpoint="/api/v5/market/candles",
            bar=bar, limit=limit, after=after, before=before,
        )

    def _history_candles(self, bar: str = "1m", limit: int = 100,
                         after=None, before=None) -> list:
        return self._request_candles(
            endpoint="/api/v5/market/history-candles",
            bar=bar, limit=limit, after=after, before=before,
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

    @staticmethod
    def _normalize_candle(candle: list) -> MarketTrade:
        ts = int(candle[0])
        return MarketTrade(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            open=float(candle[1]),
            high=float(candle[2]),
            low=float(candle[3]),
            close=float(candle[4]),
            volume=float(candle[5]),
        )

    def stream_history(self, start, end, step: str):
        start_ms = _to_epoch_ms(start)
        end_ms = _to_epoch_ms(end)
        if end_ms < start_ms:
            raise ValueError("backtest.end must be greater than backtest.start")

        all_candles = []
        cursor = str(end_ms)
        while True:
            raw = self._history_candles(
                bar=step, limit=100,
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
                yield self._normalize_candle(c)

    def stream_prices(self, step: str):
        last_ts = None
        while True:
            raw = self._candles(bar=step, limit=2)
            for candle in raw:
                ts = int(candle[0])
                confirmed = len(candle) > 8 and str(candle[8]) == "1"
                if confirmed and ts != last_ts:
                    last_ts = ts
                    yield self._normalize_candle(candle)
            time.sleep(1.0)

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
            self._account.set_leverage(self._instrument, self._leverage)
            self._leverage_set = True

        side = position.side
        usd_size = position.size
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            contracts = self._usd_to_contracts(usd_size)
            if contracts <= 0:
                return OpenResult(
                    success=False,
                    message=f"size too small usd={usd_size:.4f} contracts={contracts}",
                )
            try:
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
