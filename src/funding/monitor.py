from __future__ import annotations

from datetime import datetime, timezone

from src.app.config import FundingConfig
from src.app.logger import AppLogger
from src.clickhouse.client import ClickHouseClient
from src.funding.execution import FundingPosition
from src.funding.repo import FundingRepo
from src.instrument.repo import InstrumentRepo


class FundingMonitor:
    def __init__(
        self,
        funding: FundingRepo,
        instruments: InstrumentRepo,
        config: FundingConfig,
        clickhouse: ClickHouseClient | None,
        logger: AppLogger,
    ):
        self._funding = funding
        self._instruments = instruments
        self._config = config
        self._ch = clickhouse
        self._logger = logger

    def run(self, positions: list[FundingPosition]) -> None:
        for pos in positions:
            self._monitor_one(pos)

    def _monitor_one(self, pos: FundingPosition) -> None:
        try:
            prices = self._instruments.get_price_pair(pos.spot_inst_id, pos.perp_inst_id)
            rate_data = self._funding.get_rate(pos.perp_inst_id)
        except Exception as exc:
            self._logger.error(f"monitor: fetch failed pair={pos.pair}: {exc}")
            return

        spot_notional = pos.spot_qty * prices.spot_price
        perp_notional = pos.perp_qty * prices.perp_price
        mid = (spot_notional + perp_notional) / 2
        drift = abs(spot_notional - perp_notional)

        self.report(pos, spot_notional, perp_notional, drift, rate_data.rate)
        self.alert(pos, drift, mid, rate_data.rate)

    def report(
        self,
        pos: FundingPosition,
        spot_notional: float,
        perp_notional: float,
        drift: float,
        current_rate: float,
    ) -> None:
        self._logger.info(
            f"monitor: {pos.pair} dir={pos.direction} "
            f"spot_n={spot_notional:.2f} perp_n={perp_notional:.2f} "
            f"drift={drift:.2f} rate={current_rate:.6f}"
        )
        if self._ch is None:
            return
        self._ch.write("funding_monitor", {
            "pair": pos.pair,
            "direction": pos.direction,
            "spot_notional": spot_notional,
            "perp_notional": perp_notional,
            "drift": drift,
            "current_funding_rate": current_rate,
            "timestamp": _now_ms(),
        })

    def alert(
        self,
        pos: FundingPosition,
        drift: float,
        mid_notional: float,
        current_rate: float,
    ) -> None:
        band = self._config.drift_band * mid_notional
        if drift > 2 * band:
            self._logger.warning(
                f"ALERT: {pos.pair} drift={drift:.2f} exceeds 2x band={band:.2f}"
            )

        if pos.direction == "long_spot" and current_rate < 0:
            self._logger.warning(f"ALERT: {pos.pair} funding rate flipped ({current_rate:.6f})")
        elif pos.direction == "short_spot" and current_rate > 0:
            self._logger.warning(f"ALERT: {pos.pair} funding rate flipped ({current_rate:.6f})")

        one_legged = (pos.spot_qty <= 0) != (pos.perp_qty <= 0)
        if one_legged:
            self._logger.warning(
                f"ALERT: {pos.pair} one-legged position "
                f"spot_qty={pos.spot_qty:.6f} perp_qty={pos.perp_qty:.6f}"
            )


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
