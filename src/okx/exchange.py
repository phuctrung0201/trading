import time
from datetime import datetime, timedelta, timezone

from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import MarketTrade, Position, OpenResult
from src.exchange.position import PositionTracker
from src.asset.repo import AssetRepo
from src.instrument.repo import InstrumentRepo
from src.trade.repo import TradeRepo


class OkxExchange(ExchangeAdapter):
    MAX_RETRIES = 5
    MARGIN_REDUCE = 0.8

    def __init__(self, okx_client, logger=None):
        self._instruments: InstrumentRepo = okx_client.instruments
        self._trades: TradeRepo = okx_client.trades
        self._assets: AssetRepo = okx_client.assets
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
        initial_equity = float(self._assets.get_balance(currency="USDT"))
        self._tracker = PositionTracker(initial_equity)

    @staticmethod
    def _normalize_trade(raw: dict) -> MarketTrade:
        return MarketTrade(
            trade_id=raw["tradeId"],
            timestamp=raw["ts"],
            price=float(raw["px"]),
            size=float(raw["sz"]),
            side=raw["side"],
        )

    def _log(self, msg: str):
        if self._logger is not None:
            self._logger.info(msg)

    def stream_history(self, depth_sec: int):
        start_ms = int(
            (datetime.now(timezone.utc) - timedelta(seconds=depth_sec)).timestamp() * 1000
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        yield from self._stream_range(start_ms, now_ms)

    def stream_range(self, start_ms: int, end_ms: int):
        yield from self._stream_range(start_ms, end_ms)

    def _stream_range(self, start_ms: int, end_ms: int):
        self._log(f"stream_range inst={self._instrument} start={start_ms} end={end_ms}")
        raw = self._trades.stream_range(self._instrument, start_ms, end_ms)
        self._log(f"stream_range fetched {len(raw)} unique trades")
        for t in raw:
            yield self._normalize_trade(t)

    def stream_prices(self):
        last_trade_id: str | None = None
        while True:
            try:
                raw = self._trades.recent_trades(self._instrument, limit=100)
                raw.sort(key=lambda t: int(t["ts"]))
                for t in raw:
                    tid = t["tradeId"]
                    if last_trade_id is not None and int(tid) <= int(last_trade_id):
                        continue
                    last_trade_id = tid
                    yield self._normalize_trade(t)
            except Exception as exc:
                if self._logger is not None:
                    self._logger.warning(
                        f"stream_prices failed instrument={self._instrument} "
                        f"last_trade_id={last_trade_id}: {exc}",
                        exc_info=True,
                    )
                time.sleep(1)
            time.sleep(0.2)

    def _ensure_contract_info(self):
        if self._ct_val is None:
            info = self._instruments.get_instrument(self._instrument)
            self._ct_val = float(info["ctVal"])
            self._lot_sz = float(info["lotSz"])

    def _usd_to_contracts(self, usd_size: float) -> float:
        self._ensure_contract_info()
        assert self._ct_val is not None and self._lot_sz is not None
        mark_price = self._instruments.get_mark_price(self._instrument)
        contracts = usd_size / (self._ct_val * mark_price)
        lots = int(contracts / self._lot_sz) * self._lot_sz
        return round(lots, 8)

    @staticmethod
    def _is_margin_error(message: str) -> bool:
        return "Insufficient" in message and "margin" in message

    def set_price(self, price: float):
        self._tracker.set_price(price)

    def fetch_asset(self, asset: str) -> float:
        return self._assets.get_balance(currency=asset)

    def fetch_position(self) -> Position | None:
        data = self._assets.get_positions(self._instrument)
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
                self._assets.set_leverage(self._instrument, self._leverage)
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
                data = self._trades.place_order(
                    instrument=self._instrument,
                    side=side,
                    size=contracts,
                )
                order_id = None
                if data and isinstance(data, list) and len(data) > 0:
                    order_id = data[0].get("ordId")

                fill_price = None
                if order_id:
                    order_detail = self._trades.get_order(self._instrument, order_id)
                    avg_px = order_detail.get("avgPx")
                    if avg_px and avg_px != "":
                        fill_price = float(avg_px)

                filled = Position(side=side, size=usd_size, price=fill_price)
                self._tracker.open(filled)
                return OpenResult(success=True, position=filled)
            except Exception as exc:
                last_error = str(exc)
                if self._logger is not None:
                    self._logger.warning(
                        f"open retry attempt={attempt+1}/{self.MAX_RETRIES} "
                        f"instrument={self._instrument} side={side} "
                        f"usd_size={usd_size:.4f}: {exc}",
                        exc_info=True,
                    )
                if self._is_margin_error(last_error):
                    usd_size *= self.MARGIN_REDUCE
                delay = min(30.0, 1.0 * (2 ** attempt))
                time.sleep(delay)

        return OpenResult(success=False, message=last_error)

    def close(self, position: Position) -> bool:
        for attempt in range(self.MAX_RETRIES):
            try:
                self._trades.close_position(instrument=self._instrument)
                self._tracker.close()
                return True
            except Exception as exc:
                if self._logger is not None:
                    self._logger.warning(
                        f"close retry attempt={attempt+1}/{self.MAX_RETRIES} "
                        f"instrument={self._instrument} side={position.side}: {exc}",
                        exc_info=True,
                    )
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
