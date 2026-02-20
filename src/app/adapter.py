import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Position:
    side: str
    size: float
    price: float | None = None


@dataclass
class OpenResult:
    success: bool
    position: Position | None = None
    message: str | None = None


class ExchangeAdapter(ABC):
    @abstractmethod
    def open(self, position: Position) -> OpenResult:
        raise NotImplementedError

    @abstractmethod
    def close(self, position: Position) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_asset(self, asset: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_position(self) -> Position | None:
        raise NotImplementedError

    @abstractmethod
    def get_equity(self) -> float:
        raise NotImplementedError


class Measurable(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class MeasurementAdapter:
    @abstractmethod
    def record(self, measurable: Measurable):
        raise NotImplementedError


class SimulateAdapter(ExchangeAdapter):
    def __init__(self, initial_equity: float = 100.0):
        self.asset = initial_equity
        self.position: Position | None = None
        self._last_price: float = 0.0
        self._equity_at_open: float = 0.0

    def set_price(self, price: float):
        self._last_price = price

    def get_equity(self) -> float:
        return max(0.0, self.asset + self._calculate_unrealized_pnl())

    def _calculate_unrealized_pnl(self) -> float:
        if self.position is None:
            return 0.0
        if self.position.price is None or self.position.price <= 0:
            return 0.0
        if self._last_price <= 0:
            return 0.0
        price_return = (self._last_price - self.position.price) / self.position.price
        direction = 1.0 if self.position.side == "buy" else -1.0
        return self.position.size * price_return * direction

    def open(self, position: Position) -> OpenResult:
        self._equity_at_open = self.get_equity()
        fill_price = position.price if position.price is not None else self._last_price
        filled = Position(side=position.side, size=position.size, price=fill_price)
        self.position = filled
        return OpenResult(success=True, position=filled, message=None)

    def close(self, position: Position) -> bool:
        if self.position is None:
            return False
        self.asset += self._calculate_unrealized_pnl()
        self.position = None
        self._equity_at_open = 0.0
        return True

    def get_asset(self, asset: str) -> float:
        _ = asset
        return self.asset

    def get_position(self) -> Position | None:
        return self.position


class OkxExchangeAdapter(SimulateAdapter):
    MAX_RETRIES = 3
    MARGIN_REDUCE = 0.8

    def __init__(self, okx_client, instrument, leverage):
        initial_equity = float(okx_client.get_asset(currency="USDT"))
        super().__init__(initial_equity=initial_equity)
        self._okx = okx_client
        self._instrument = instrument
        self._leverage = leverage
        self._leverage_set = False
        self._ct_val: float | None = None
        self._lot_sz: float | None = None

    def _ensure_contract_info(self):
        if self._ct_val is None:
            info = self._okx.get_instrument(self._instrument)
            self._ct_val = float(info["ctVal"])
            self._lot_sz = float(info["lotSz"])

    def _usd_to_contracts(self, usd_size: float) -> float:
        self._ensure_contract_info()
        assert self._ct_val is not None and self._lot_sz is not None
        mark_price = self._okx.get_mark_price(self._instrument)
        contracts = usd_size / (self._ct_val * mark_price)
        lots = int(contracts / self._lot_sz) * self._lot_sz
        return round(lots, 8)

    def _is_margin_error(self, message: str) -> bool:
        return "Insufficient" in message and "margin" in message

    def fetch_asset(self, asset: str) -> float:
        return self._okx.get_asset(currency=asset)

    def fetch_position(self) -> Position | None:
        data = self._okx.get_positions(self._instrument)
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
            self._okx.set_leverage(self._instrument, self._leverage)
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
                data = self._okx.place_order(
                    instrument=self._instrument,
                    side=side,
                    size=contracts,
                )
                order_id = None
                if data and isinstance(data, list) and len(data) > 0:
                    order_id = data[0].get("ordId")

                fill_price = None
                if order_id:
                    order_detail = self._okx.get_order(self._instrument, order_id)
                    avg_px = order_detail.get("avgPx")
                    if avg_px and avg_px != "":
                        fill_price = float(avg_px)

                filled = Position(side=side, size=usd_size, price=fill_price)
                super().open(filled)
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
                self._okx.close_position(
                    instrument=self._instrument,
                )
                super().close(position)
                return True
            except Exception:
                delay = min(30.0, 1.0 * (2 ** attempt))
                time.sleep(delay)
        return False


class ClickHouseAdapter(MeasurementAdapter):
    def __init__(self, clickhouse_client, session_id: str,
                 setup_name: str | None = None,
                 instrument: str | None = None,
                 strategy: str | None = None):
        self.clickhouse_client = clickhouse_client
        self.session_id = session_id
        self.setup_name = setup_name or ""
        self.instrument = instrument or ""
        self.strategy = strategy or ""

    def _table_for(self, measurable) -> str:
        from src.measurement.ops import OpsMeasurement
        if isinstance(measurable, OpsMeasurement):
            return "ops"
        return "trade_event"

    def record(self, measurable):
        fields = measurable.to_dict()
        timestamp = getattr(measurable, "timestamp", None)
        row = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "setup": self.setup_name,
            "instrument": self.instrument,
            "strategy": self.strategy,
            **fields,
        }
        table = self._table_for(measurable)
        if self.clickhouse_client is not None:
            self.clickhouse_client.write(table, row)
        return row
