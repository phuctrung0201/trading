from __future__ import annotations

from dataclasses import dataclass

from src.app.config import FundingConfig
from src.app.logger import AppLogger
from src.clickhouse.client import ClickHouseClient
from src.clickhouse.recorder import Recorder
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import FundingSnapshot, MarketTrade, Position
from src.instrument.universe import Universe
from src.strategy.adapter import SignalResult, StrategyAdapter


@dataclass
class SimPosition:
    pair: str
    direction: str
    spot_qty: float
    perp_qty: float
    spot_entry_price: float
    perp_entry_price: float
    entry_rate: float


@dataclass
class BacktestResult:
    total_pnl: float
    funding_earned: float
    rebalance_count: int
    max_drawdown: float
    steps: int
    positions_opened: int
    positions_closed: int


class FundingStrategy(StrategyAdapter):
    """Multi-instrument funding capture strategy on 8h snapshots.

    Receives a universe of instrument IDs at bootstrap and manages
    independent positions per instrument.  Each position gets an equal
    share of the configured notional (``notional / universe_size``).
    """

    def __init__(self, recorder: Recorder, logger: AppLogger,
                 clickhouse: ClickHouseClient | None = None):
        super().__init__(recorder=recorder, logger=logger)
        self._ch = clickhouse
        self._config: FundingConfig | None = None
        self._universe: Universe = Universe([])
        self._positions: dict[str, SimPosition] = {}
        self._hold_counts: dict[str, int] = {}
        self._funding_earned: float = 0.0
        self._rebalance_count: int = 0
        self._step_count: int = 0
        self._positions_opened: int = 0
        self._positions_closed: int = 0
        self._max_drawdown: float = 0.0
        self._initial_equity: float = 0.0

    def bootstrap(
        self,
        exchange: ExchangeAdapter,
        config: FundingConfig,
        universe: Universe,
    ):
        super().bootstrap(exchange)
        self._config = config
        self._fee_rate = config.fee_rate
        self._initial_equity = config.notional
        self._universe = universe

    def ack_trade(self, trade: MarketTrade):
        raise NotImplementedError("Use ack_funding for funding strategies")

    def ack_funding(self, snapshot: FundingSnapshot) -> None:
        self._step_count += 1
        trade = self._snapshot_as_trade(snapshot)
        inst_id = snapshot.inst_id

        self.exchange.set_price(snapshot.spot_price)
        self._mark_to_market()

        pos = self._positions.get(inst_id)

        if pos is not None:
            self._collect_funding(pos, snapshot, trade)
            self._hold_counts[inst_id] = self._hold_counts.get(inst_id, 0) + 1

        if pos is not None:
            if self._should_exit(pos, snapshot):
                self._exit(pos, inst_id, snapshot, trade)
            else:
                self._manage(pos, snapshot, trade)
        elif self._should_enter(snapshot):
            self._enter(inst_id, snapshot, trade)

        pos = self._positions.get(inst_id)
        if pos is not None:
            self._write_monitor(pos, snapshot)

        self._sync_aggregate_position()
        self._track_max_drawdown()
        self._emit_trade_measurement(trade, SignalResult())

    def results(self) -> BacktestResult:
        pnl = self.exchange.get_equity() - self._initial_equity
        return BacktestResult(
            total_pnl=pnl,
            funding_earned=self._funding_earned,
            rebalance_count=self._rebalance_count,
            max_drawdown=self._max_drawdown,
            steps=self._step_count,
            positions_opened=self._positions_opened,
            positions_closed=self._positions_closed,
        )

    def _per_position_notional(self) -> float:
        return self._config.notional / max(len(self._universe), 1)

    def _snapshot_as_trade(self, snap: FundingSnapshot) -> MarketTrade:
        return MarketTrade(
            trade_id=f"{snap.inst_id}:{snap.timestamp}",
            timestamp=str(snap.timestamp),
            price=snap.spot_price,
            size=0.0,
            side="",
        )

    def _should_enter(self, snap: FundingSnapshot) -> bool:
        return abs(snap.funding_rate) >= self._config.min_funding_rate

    def _enter(self, inst_id: str, snap: FundingSnapshot, trade: MarketTrade) -> None:
        direction = "long_spot" if snap.funding_rate > 0 else "short_spot"
        notional = self._per_position_notional()
        spot_qty = notional / snap.spot_price
        perp_qty = notional / snap.perp_price

        fee = notional * self._fee_rate * 2
        self.exchange.adjust_equity(-fee)

        self._positions[inst_id] = SimPosition(
            pair=inst_id,
            direction=direction,
            spot_qty=spot_qty,
            perp_qty=perp_qty,
            spot_entry_price=snap.spot_price,
            perp_entry_price=snap.perp_price,
            entry_rate=snap.funding_rate,
        )
        self._hold_counts[inst_id] = 0
        self._positions_opened += 1

        pair = inst_id.split("-")[0] if "-" in inst_id else inst_id
        self._write_screen(pair, inst_id, direction, snap.funding_rate, snap.timestamp)

        self._logger.info(
            f"enter {inst_id} dir={direction} notional={notional:.2f} "
            f"rate={snap.funding_rate:.6f}"
        )
        self._emit_event(
            trade, "open",
            fill_price=snap.spot_price,
            signal=direction,
            reason=f"{inst_id} funding rate {snap.funding_rate:.6f}",
            fee=fee,
        )

    def _collect_funding(
        self, pos: SimPosition, snap: FundingSnapshot, trade: MarketTrade,
    ) -> None:
        perp_notional = pos.perp_qty * snap.perp_price
        if pos.direction == "long_spot":
            payment = perp_notional * snap.funding_rate
        else:
            payment = perp_notional * (-snap.funding_rate)
        self._funding_earned += payment
        self.exchange.adjust_equity(payment)

    def _should_exit(self, pos: SimPosition, snap: FundingSnapshot) -> bool:
        hold = self._hold_counts.get(pos.pair, 0)
        if hold < self._config.min_hold_periods:
            return False
        if pos.direction == "long_spot" and snap.funding_rate < 0:
            return True
        if pos.direction == "short_spot" and snap.funding_rate > 0:
            return True
        exit_rate = self._config.exit_funding_rate or self._config.min_funding_rate
        if abs(snap.funding_rate) < exit_rate:
            return True
        return False

    def _exit(
        self, pos: SimPosition, inst_id: str,
        snap: FundingSnapshot, trade: MarketTrade,
    ) -> None:
        spot_pnl = pos.spot_qty * (snap.spot_price - pos.spot_entry_price)
        perp_pnl = pos.perp_qty * (pos.perp_entry_price - snap.perp_price)
        if pos.direction == "short_spot":
            spot_pnl = -spot_pnl
            perp_pnl = -perp_pnl

        close_notional = pos.spot_qty * snap.spot_price + pos.perp_qty * snap.perp_price
        fee = close_notional * self._fee_rate
        basis_pnl = spot_pnl + perp_pnl
        self.exchange.adjust_equity(basis_pnl - fee)

        self._positions_closed += 1
        del self._positions[inst_id]
        self._hold_counts.pop(inst_id, None)
        self._logger.info(
            f"exit {inst_id} basis_pnl={basis_pnl:.2f} fee={fee:.2f} "
            f"equity={self.exchange.get_equity():.2f}"
        )
        self._emit_event(
            trade, "close",
            fill_price=snap.spot_price,
            pnl=basis_pnl,
            reason=f"{inst_id} funding rate {snap.funding_rate:.6f}",
            fee=fee,
        )

    def _manage(
        self, pos: SimPosition, snap: FundingSnapshot, trade: MarketTrade,
    ) -> None:
        spot_notional = pos.spot_qty * snap.spot_price
        perp_notional = pos.perp_qty * snap.perp_price
        mid = (spot_notional + perp_notional) / 2
        drift = abs(spot_notional - perp_notional)
        band = self._config.drift_band * mid

        if drift <= band:
            return

        new_spot_qty = mid / snap.spot_price
        new_perp_qty = mid / snap.perp_price
        rebal_notional = abs(new_spot_qty - pos.spot_qty) * snap.spot_price
        fee = rebal_notional * self._fee_rate * 2
        self.exchange.adjust_equity(-fee)

        pos.spot_qty = new_spot_qty
        pos.perp_qty = new_perp_qty
        self._rebalance_count += 1
        self._logger.info(
            f"rebalance {pos.pair} drift={drift:.2f} band={band:.2f} fee={fee:.4f}"
        )
        self._emit_event(
            trade, "resize",
            fill_price=snap.spot_price,
            reason=f"rebalance {pos.pair} drift={drift:.2f}",
            fee=fee,
        )

    def _sync_aggregate_position(self) -> None:
        if not self._positions:
            self._current_position = None
            return
        total = sum(
            p.spot_qty * p.spot_entry_price + p.perp_qty * p.perp_entry_price
            for p in self._positions.values()
        )
        if self._current_position is None:
            self._current_position = Position(side="buy", size=total)
        self._current_position.size = total

    def _track_max_drawdown(self) -> None:
        dd = self._calculate_drawdown()
        if dd < self._max_drawdown:
            self._max_drawdown = dd

    def _write_screen(self, pair: str, inst_id: str, direction: str,
                      funding_rate: float, timestamp: int) -> None:
        if self._ch is None:
            return
        self._ch.write("funding_screen", {
            "session_id": self.recorder.session_id,
            "pair": pair,
            "inst_id": inst_id,
            "direction": direction,
            "funding_rate": funding_rate,
            "timestamp": timestamp,
        })

    def _write_monitor(self, pos: SimPosition, snap: FundingSnapshot) -> None:
        if self._ch is None:
            return
        spot_notional = pos.spot_qty * snap.spot_price
        perp_notional = pos.perp_qty * snap.perp_price
        drift = abs(spot_notional - perp_notional)
        self._ch.write("funding_monitor", {
            "pair": pos.pair,
            "direction": pos.direction,
            "spot_notional": spot_notional,
            "perp_notional": perp_notional,
            "drift": drift,
            "current_funding_rate": snap.funding_rate,
            "timestamp": snap.timestamp,
        })
