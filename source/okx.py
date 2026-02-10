"""OKX exchange client for demo and live trading via the REST API v5."""

import asyncio
import base64
import hashlib
import hmac
import json
import queue
import threading
from datetime import datetime, timezone
from dataclasses import dataclass

import requests
import websockets

_BASE_URL = "https://www.okx.com"
_WS_BUSINESS_URL = "wss://ws.okx.com/ws/v5/business"
_WS_DEMO_BUSINESS_URL = "wss://wspap.okx.com:8443/ws/v5/business?brokerId=9999"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Represents a placed or queried order."""

    order_id: str
    client_order_id: str
    instrument: str
    side: str
    size: str
    price: str | None
    order_type: str
    state: str

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, instrument={self.instrument}, "
            f"side={self.side}, size={self.size}, price={self.price}, "
            f"type={self.order_type}, state={self.state})"
        )


@dataclass
class Position:
    """Represents an open position."""

    instrument: str
    side: str
    size: str
    avg_price: str
    unrealised_pnl: str
    leverage: str

    def __repr__(self) -> str:
        return (
            f"Position(instrument={self.instrument}, side={self.side}, "
            f"size={self.size}, avg_price={self.avg_price}, "
            f"upnl={self.unrealised_pnl}, leverage={self.leverage})"
        )


@dataclass
class Candle:
    """Represents a single OHLCV candle from the WebSocket feed."""

    timestamp: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    volume_currency: str
    volume_quote: str
    confirm: bool

    def __repr__(self) -> str:
        status = "closed" if self.confirm else "open"
        return (
            f"{self.timestamp}  O={self.open}  H={self.high}  "
            f"L={self.low}  C={self.close}  V={self.volume}  [{status}]"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OKXError(Exception):
    """Raised when the OKX API returns a non-zero error code."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"OKX error {code}: {message}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class Client:
    """OKX REST API v5 client.

    Parameters
    ----------
    API_KEY : str
        OKX API key.
    SECRET_KEY : str
        OKX secret key.
    PASSPHRASE : str
        OKX API passphrase (set when creating the API key).
    demo : bool
        If ``True``, send the ``x-simulated-trading: 1`` header so all
        requests hit the demo/paper-trading environment.
    """

    def __init__(
        self,
        API_KEY: str,
        SECRET_KEY: str,
        PASSPHRASE: str = "",
        demo: bool = True,
    ):
        self._api_key = API_KEY
        self._secret_key = SECRET_KEY
        self._passphrase = PASSPHRASE
        self._demo = demo
        self._base_url = _BASE_URL
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def _timestamp(self) -> str:
        """Return the current UTC timestamp in ISO-8601 format for OKX."""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Create the HMAC-SHA256 signature required by OKX.

        Signature = Base64(HMAC-SHA256(timestamp + METHOD + requestPath + body, secretKey))
        """
        prehash = timestamp + method.upper() + path + body
        mac = hmac.new(
            self._secret_key.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        """Build authenticated request headers."""
        ts = self._timestamp()
        headers = {
            "OK-ACCESS-KEY": self._api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type": "application/json",
        }
        if self._demo:
            headers["x-simulated-trading"] = "1"
        return headers

    # ------------------------------------------------------------------
    # Low-level request helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Authenticated GET request."""
        if params:
            query = "&".join(
                f"{k}={v}" for k, v in params.items() if v is not None
            )
            full_path = f"{path}?{query}" if query else path
        else:
            full_path = path

        headers = self._headers("GET", full_path)
        resp = self._session.get(
            self._base_url + full_path, headers=headers, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise OKXError(data.get("code", "?"), data.get("msg", "unknown error"))
        return data

    def _post(self, path: str, body: dict | None = None) -> dict:
        """Authenticated POST request."""
        body_str = json.dumps(body) if body else ""
        headers = self._headers("POST", path, body_str)
        resp = self._session.post(
            self._base_url + path, headers=headers, data=body_str, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise OKXError(data.get("code", "?"), data.get("msg", "unknown error"))
        return data

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def balance(self, currency: str | None = None) -> list[dict]:
        """Get account balance.

        Parameters
        ----------
        currency : str, optional
            Filter by currency, e.g. ``"USDT"``.

        Returns
        -------
        list[dict]
            List of balance detail dictionaries from the OKX API.
        """
        params = {}
        if currency:
            params["ccy"] = currency.upper()
        data = self._get("/api/v5/account/balance", params or None)
        return data["data"]

    def positions(self, instrument: str | None = None) -> list[Position]:
        """Get open positions.

        Parameters
        ----------
        instrument : str, optional
            Instrument ID, e.g. ``"ETH-USDT-SWAP"``.

        Returns
        -------
        list[Position]
        """
        params = {}
        if instrument:
            params["instId"] = instrument
        data = self._get("/api/v5/account/positions", params or None)
        return [
            Position(
                instrument=p["instId"],
                side=p.get("posSide", "net"),
                size=p.get("pos", "0"),
                avg_price=p.get("avgPx", "0"),
                unrealised_pnl=p.get("upl", "0"),
                leverage=p.get("lever", "0"),
            )
            for p in data["data"]
        ]

    def set_leverage(
        self,
        instrument: str,
        leverage: str,
        margin_mode: str = "cross",
        position_side: str | None = None,
    ) -> dict:
        """Set leverage for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument ID, e.g. ``"ETH-USDT-SWAP"``.
        leverage : str
            Leverage value, e.g. ``"10"``.
        margin_mode : str
            ``"cross"`` or ``"isolated"``.
        position_side : str, optional
            ``"long"`` or ``"short"`` (required in long/short mode).

        Returns
        -------
        dict
        """
        body: dict = {
            "instId": instrument,
            "lever": leverage,
            "mgnMode": margin_mode,
        }
        if position_side:
            body["posSide"] = position_side
        data = self._post("/api/v5/account/set-leverage", body)
        return data["data"][0] if data["data"] else {}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def ticker(self, instrument: str) -> dict:
        """Get the latest ticker for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument ID, e.g. ``"ETH-USDT"`` or ``"BTC-USDT-SWAP"``.

        Returns
        -------
        dict
            Ticker data including last price, bid/ask, 24h volume, etc.
        """
        data = self._get("/api/v5/market/ticker", {"instId": instrument})
        return data["data"][0] if data["data"] else {}

    def orderbook(self, instrument: str, depth: int = 20) -> dict:
        """Get order book for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument ID, e.g. ``"ETH-USDT"``.
        depth : int
            Number of levels (1–400).

        Returns
        -------
        dict
            Contains ``asks`` and ``bids`` lists.
        """
        data = self._get(
            "/api/v5/market/books", {"instId": instrument, "sz": str(depth)},
        )
        return data["data"][0] if data["data"] else {}

    def candles(
        self,
        instrument: str,
        bar: str = "1H",
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
    ) -> list[list]:
        """Get candlestick (OHLCV) data.

        Parameters
        ----------
        instrument : str
            Instrument ID, e.g. ``"ETH-USDT"``.
        bar : str
            Bar size, e.g. ``"1m"``, ``"5m"``, ``"1H"``, ``"1D"``.
        limit : int
            Number of candles (max 300).
        after : str, optional
            Pagination — return records earlier than this timestamp (ms).
        before : str, optional
            Pagination — return records newer than this timestamp (ms).

        Returns
        -------
        list[list]
            Each inner list: ``[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]``.
        """
        params: dict = {"instId": instrument, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        data = self._get("/api/v5/market/candles", params)
        return data["data"]

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def place_order(
        self,
        instrument: str,
        side: str,
        size: str,
        order_type: str = "market",
        price: str | None = None,
        trade_mode: str = "cross",
        target_currency: str = "base_ccy",
        client_order_id: str | None = None,
    ) -> Order:
        """Place a new order.

        Parameters
        ----------
        instrument : str
            Instrument ID, e.g. ``"ETH-USDT"`` (spot) or
            ``"ETH-USDT-SWAP"`` (perpetual).
        side : str
            ``"buy"`` or ``"sell"``.
        size : str
            Order size (in base or quote currency depending on *target_currency*).
        order_type : str
            ``"market"``, ``"limit"``, ``"post_only"``, ``"fok"``, ``"ioc"``.
        price : str, optional
            Required for limit orders.
        trade_mode : str
            ``"cross"``, ``"isolated"``, or ``"cash"`` (spot without margin).
        target_currency : str
            ``"base_ccy"`` or ``"quote_ccy"`` — which currency *size* is
            denominated in.
        client_order_id : str, optional
            Custom order ID for tracking.

        Returns
        -------
        Order
        """
        body: dict = {
            "instId": instrument,
            "tdMode": trade_mode,
            "side": side,
            "ordType": order_type,
            "sz": size,
            "tgtCcy": target_currency,
        }
        if price is not None:
            body["px"] = price
        if client_order_id:
            body["clOrdId"] = client_order_id

        data = self._post("/api/v5/trade/order", body)
        info = data["data"][0]
        return Order(
            order_id=info.get("ordId", ""),
            client_order_id=info.get("clOrdId", ""),
            instrument=instrument,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            state="submitted",
        )

    def cancel_order(self, instrument: str, order_id: str) -> dict:
        """Cancel an open order.

        Parameters
        ----------
        instrument : str
            Instrument ID.
        order_id : str
            Order ID returned from :meth:`place_order`.

        Returns
        -------
        dict
        """
        body = {"instId": instrument, "ordId": order_id}
        data = self._post("/api/v5/trade/cancel-order", body)
        return data["data"][0]

    def get_order(self, instrument: str, order_id: str) -> Order:
        """Query an order by ID.

        Parameters
        ----------
        instrument : str
            Instrument ID.
        order_id : str
            Order ID.

        Returns
        -------
        Order
        """
        data = self._get(
            "/api/v5/trade/order", {"instId": instrument, "ordId": order_id},
        )
        info = data["data"][0]
        return Order(
            order_id=info.get("ordId", ""),
            client_order_id=info.get("clOrdId", ""),
            instrument=info.get("instId", instrument),
            side=info.get("side", ""),
            size=info.get("sz", ""),
            price=info.get("px", None),
            order_type=info.get("ordType", ""),
            state=info.get("state", ""),
        )

    def pending_orders(self, instrument: str | None = None) -> list[Order]:
        """Get all pending (open) orders.

        Parameters
        ----------
        instrument : str, optional
            Filter by instrument ID.

        Returns
        -------
        list[Order]
        """
        params = {}
        if instrument:
            params["instId"] = instrument
        data = self._get("/api/v5/trade/orders-pending", params or None)
        return [
            Order(
                order_id=o.get("ordId", ""),
                client_order_id=o.get("clOrdId", ""),
                instrument=o.get("instId", ""),
                side=o.get("side", ""),
                size=o.get("sz", ""),
                price=o.get("px", None),
                order_type=o.get("ordType", ""),
                state=o.get("state", ""),
            )
            for o in data["data"]
        ]

    def close_position(
        self,
        instrument: str,
        margin_mode: str = "cross",
        position_side: str | None = None,
    ) -> dict:
        """Close an open position.

        Parameters
        ----------
        instrument : str
            Instrument ID.
        margin_mode : str
            ``"cross"`` or ``"isolated"``.
        position_side : str, optional
            ``"long"`` or ``"short"`` (required in long/short mode).

        Returns
        -------
        dict
        """
        body: dict = {
            "instId": instrument,
            "mgnMode": margin_mode,
        }
        if position_side:
            body["posSide"] = position_side
        data = self._post("/api/v5/trade/close-position", body)
        return data["data"][0] if data["data"] else {}

    def order_history(
        self,
        instrument_type: str = "SPOT",
        instrument: str | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get recent order history (last 7 days).

        Parameters
        ----------
        instrument_type : str
            ``"SPOT"``, ``"MARGIN"``, ``"SWAP"``, ``"FUTURES"``, ``"OPTION"``.
        instrument : str, optional
            Filter by instrument ID.
        limit : int
            Max number of results (up to 100).

        Returns
        -------
        list[Order]
        """
        params: dict = {"instType": instrument_type, "limit": str(limit)}
        if instrument:
            params["instId"] = instrument
        data = self._get("/api/v5/trade/orders-history-archive", params)
        return [
            Order(
                order_id=o.get("ordId", ""),
                client_order_id=o.get("clOrdId", ""),
                instrument=o.get("instId", ""),
                side=o.get("side", ""),
                size=o.get("sz", ""),
                price=o.get("px", None),
                order_type=o.get("ordType", ""),
                state=o.get("state", ""),
            )
            for o in data["data"]
        ]

    # ------------------------------------------------------------------
    # WebSocket – real-time candle subscription
    # ------------------------------------------------------------------

    def subscribe(self, instrument: str, bar: str = "1m") -> "CandleChannel":
        """Subscribe to real-time OHLCV candles via WebSocket.

        Returns a :class:`CandleChannel` that can be iterated with a
        ``for`` loop.  The WebSocket runs in a background thread so the
        caller can consume candles synchronously::

            channel = client.subscribe("ETH-USDT", bar="1m")
            for candle in channel:
                print(candle)

        Parameters
        ----------
        instrument : str
            Instrument ID, e.g. ``"ETH-USDT"`` or ``"BTC-USDT-SWAP"``.
        bar : str
            Candle period, e.g. ``"1m"``, ``"5m"``, ``"15m"``, ``"1H"``,
            ``"1D"``.  Translates to the OKX channel ``candle{bar}``.

        Returns
        -------
        CandleChannel
            An iterator that yields :class:`Candle` objects.
        """
        # Market data (candles) is identical on live and demo feeds.
        # The demo endpoint uses port 8443 which is often blocked, so
        # always use the live business endpoint for public market data.
        return CandleChannel(_WS_BUSINESS_URL, instrument, bar)


# ---------------------------------------------------------------------------
# CandleChannel – synchronous iterator backed by a WebSocket thread
# ---------------------------------------------------------------------------

_SENTINEL = object()


class CandleChannel:
    """Iterable stream of real-time :class:`Candle` updates.

    Internally runs the async WebSocket connection on a daemon thread
    and feeds candles through a :class:`queue.Queue` so the caller can
    consume them with a plain ``for`` loop.

    The channel reconnects automatically if the connection drops.
    Call :meth:`close` or break out of the loop to stop.
    """

    def __init__(self, ws_url: str, instrument: str, bar: str) -> None:
        self._ws_url = ws_url
        self._instrument = instrument
        self._bar = bar
        self._channel = f"candle{bar}"
        self._queue: queue.Queue[Candle | object] = queue.Queue()
        self._stopped = threading.Event()

        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"ws-{instrument}-{bar}",
        )
        self._thread.start()

    # -- iterator protocol -------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> Candle:
        while True:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                if self._stopped.is_set():
                    raise StopIteration
                continue
            if item is _SENTINEL:
                raise StopIteration
            return item

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Signal the background thread to stop."""
        self._stopped.set()
        self._queue.put(_SENTINEL)

    # -- background thread -------------------------------------------------

    def _run(self) -> None:
        """Entry point for the daemon thread."""
        asyncio.run(self._ws_loop())

    async def _ws_loop(self) -> None:
        """Maintain the WebSocket connection and push candles to the queue."""
        subscribe_msg = json.dumps({
            "op": "subscribe",
            "args": [{"channel": self._channel, "instId": self._instrument}],
        })

        while not self._stopped.is_set():
            try:
                print(f"Connecting to OKX WebSocket ({self._channel} {self._instrument}) …")
                async with websockets.connect(
                    self._ws_url, ping_interval=20, open_timeout=10,
                ) as ws:
                    await ws.send(subscribe_msg)

                    resp = json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=10),
                    )
                    if resp.get("event") == "error":
                        raise OKXError(
                            resp.get("code", "?"),
                            resp.get("msg", "subscribe failed"),
                        )
                    print(f"Subscribed to {self._channel} {self._instrument}")

                    async for raw in ws:
                        if self._stopped.is_set():
                            return

                        msg = json.loads(raw)
                        if "data" not in msg:
                            continue

                        for entry in msg["data"]:
                            candle = Candle(
                                timestamp=datetime.fromtimestamp(
                                    int(entry[0]) / 1000, tz=timezone.utc,
                                ).strftime("%Y-%m-%d %H:%M:%S"),
                                open=entry[1],
                                high=entry[2],
                                low=entry[3],
                                close=entry[4],
                                volume=entry[5],
                                volume_currency=entry[6],
                                volume_quote=entry[7],
                                confirm=entry[8] == "1",
                            )
                            self._queue.put(candle)

            except (websockets.ConnectionClosed, TimeoutError):
                if not self._stopped.is_set():
                    print("WebSocket disconnected, reconnecting in 3s …")
                    await asyncio.sleep(3)

        self._queue.put(_SENTINEL)
