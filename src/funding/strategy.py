from __future__ import annotations

from dataclasses import dataclass

from src.app.config import FundingConfig
from src.app.logger import AppLogger
from src.clickhouse.recorder import Recorder
from src.exchange.adapter import ExchangeAdapter
from src.exchange.dto import FundingSnapshot, MarketTrade, Position
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


class FundingBacktestStrategy(StrategyAdapter):
    """Simulates the funding capture strategy on historical 8h snapshots.

    Extends StrategyAdapter to share the same infrastructure as other
    strategies: Recorder for ClickHouse events, SimulateExchange for
    equity tracking, and the standard event/measurement pipeline.
    """

    def __init__(self, recorder: Recorder, logger: AppLogger):
        super().__init__(recorder=recorder, logger=logger)
        self._config: FundingConfig | None = None
        self._sim_position: SimPosition | None = None
        self._funding_earned: float = 0.0
        self._rebalance_count: int = 0
        self._step_count: int = 0
        self._positions_opened: int = 0
        self._positions_closed: int = 0
        self._max_drawdown: float = 0.0
        self._fee_rate: float = 0.001
        self._initial_equity: float = 0.0

    def bootstrap(self, exchange: ExchangeAdapter, config: FundingConfig):
        super().bootstrap(exchange)
        self._config = config
        self._initial_equity = config.notional

    def ack_trade(self, trade: MarketTrade):
        raise NotImplementedError("Use ack_funding for funding strategies")

    def ack_funding(self, snapshot: FundingSnapshot) -> None:
        self._step_count += 1
        trade = self._snapshot_as_trade(snapshot)

        self.exchange.set_price(snapshot.spot_price)
        self._mark_to_market()

        if self._sim_position is not None:
            self._collect_funding(snapshot, trade)

        if self._sim_position is not None:
            if self._should_exit(snapshot):
                self._exit(snapshot, trade)
            else:
                self._manage(snapshot, trade)

        if self._sim_position is None:
            if self._should_enter(snapshot):
                self._enter(snapshot, trade)

        if self._sim_position is not None and self._current_position is not None:
            total_notional = (
                self._sim_position.spot_qty * snapshot.spot_price
                + self._sim_position.perp_qty * snapshot.perp_price
            )
            self._current_position.size = total_notional

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

    def _snapshot_as_trade(self, snap: FundingSnapshot) -> MarketTrade:
        return MarketTrade(
            trade_id=str(snap.timestamp),
            timestamp=str(snap.timestamp),
            price=snap.spot_price,
            size=0.0,
            side="",
        )

    def _should_enter(self, snap: FundingSnapshot) -> bool:
        return abs(snap.funding_rate) >= self._config.min_funding_rate

    def _enter(self, snap: FundingSnapshot, trade: MarketTrade) -> None:
        direction = "long_spot" if snap.funding_rate > 0 else "short_spot"
        notional = self.exchange.get_equity()
        spot_qty = notional / snap.spot_price
        perp_qty = notional / snap.perp_price

        fee = notional * self._fee_rate * 2
        self.exchange.adjust_equity(-fee)

        side = "buy" if direction == "long_spot" else "sell"
        self._current_position = Position(side=side, size=notional, price=snap.spot_price)

        self._sim_position = SimPosition(
            pair="SIM",
            direction=direction,
            spot_qty=spot_qty,
            perp_qty=perp_qty,
            spot_entry_price=snap.spot_price,
            perp_entry_price=snap.perp_price,
            entry_rate=snap.funding_rate,
        )
        self._positions_opened += 1
        self._logger.info(
            f"enter dir={direction} notional={notional:.2f} rate={snap.funding_rate:.6f}"
        )
        self._emit_event(
            trade, "open",
            fill_price=snap.spot_price,
            signal=direction,
            reason=f"funding rate {snap.funding_rate:.6f}",
            fee=fee,
        )

    def _collect_funding(self, snap: FundingSnapshot, trade: MarketTrade) -> None:
        pos = self._sim_position
        perp_notional = pos.perp_qty * snap.perp_price
        if pos.direction == "long_spot":
            payment = perp_notional * snap.funding_rate
        else:
            payment = perp_notional * (-snap.funding_rate)
        self._funding_earned += payment
        self.exchange.adjust_equity(payment)

    def _should_exit(self, snap: FundingSnapshot) -> bool:
        pos = self._sim_position
        if pos.direction == "long_spot" and snap.funding_rate < 0:
            return True
        if pos.direction == "short_spot" and snap.funding_rate > 0:
            return True
        if abs(snap.funding_rate) < self._config.min_funding_rate:
            return True
        return False

    def _exit(self, snap: FundingSnapshot, trade: MarketTrade) -> None:
        pos = self._sim_position
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
        self._logger.info(
            f"exit basis_pnl={basis_pnl:.2f} fee={fee:.2f} "
            f"equity={self.exchange.get_equity():.2f}"
        )
        self._emit_event(
            trade, "close",
            fill_price=snap.spot_price,
            pnl=basis_pnl,
            reason=f"funding rate {snap.funding_rate:.6f}",
            fee=fee,
        )
        self._sim_position = None
        self._current_position = None

    def _manage(self, snap: FundingSnapshot, trade: MarketTrade) -> None:
        pos = self._sim_position
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
        self._logger.info(f"rebalance drift={drift:.2f} band={band:.2f} fee={fee:.4f}")
        self._emit_event(
            trade, "resize",
            fill_price=snap.spot_price,
            reason=f"rebalance drift={drift:.2f}",
            fee=fee,
        )

    def _track_max_drawdown(self) -> None:
        dd = self._calculate_drawdown()
        if dd < self._max_drawdown:
            self._max_drawdown = dd
