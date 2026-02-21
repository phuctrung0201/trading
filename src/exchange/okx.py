import time

from src.exchange.adapter import Exchange
from src.exchange.position import PositionTracker
from src.exchange.types import Position, OpenResult


class OkxExchange(Exchange):
    """Live trading: executes on OKX, mirrors state in PositionTracker."""

    MAX_RETRIES = 3
    MARGIN_REDUCE = 0.8

    def __init__(self, okx_client, instrument: str, leverage: int = 1):
        initial_equity = float(okx_client.account.get_asset(currency="USDT"))
        self._tracker = PositionTracker(initial_equity)
        self._okx = okx_client
        self._instrument = instrument
        self._leverage = leverage
        self._leverage_set = False
        self._ct_val: float | None = None
        self._lot_sz: float | None = None

    def _ensure_contract_info(self):
        if self._ct_val is None:
            info = self._okx.market.get_instrument(self._instrument)
            self._ct_val = float(info["ctVal"])
            self._lot_sz = float(info["lotSz"])

    def _usd_to_contracts(self, usd_size: float) -> float:
        self._ensure_contract_info()
        assert self._ct_val is not None and self._lot_sz is not None
        mark_price = self._okx.market.get_mark_price(self._instrument)
        contracts = usd_size / (self._ct_val * mark_price)
        lots = int(contracts / self._lot_sz) * self._lot_sz
        return round(lots, 8)

    def _is_margin_error(self, message: str) -> bool:
        return "Insufficient" in message and "margin" in message

    def set_price(self, price: float):
        self._tracker.set_price(price)

    def fetch_asset(self, asset: str) -> float:
        return self._okx.account.get_asset(currency=asset)

    def fetch_position(self) -> Position | None:
        data = self._okx.account.get_positions(self._instrument)
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
            self._okx.account.set_leverage(self._instrument, self._leverage)
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
                data = self._okx.trading.place_order(
                    instrument=self._instrument,
                    side=side,
                    size=contracts,
                )
                order_id = None
                if data and isinstance(data, list) and len(data) > 0:
                    order_id = data[0].get("ordId")

                fill_price = None
                if order_id:
                    order_detail = self._okx.trading.get_order(self._instrument, order_id)
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
                self._okx.trading.close_position(instrument=self._instrument)
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
