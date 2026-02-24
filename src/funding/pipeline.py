from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.app.config import FundingConfig
from src.app.logger import AppLogger
from src.clickhouse.client import ClickHouseClient
from src.funding.execution import (
    FundingPosition,
    enter_position,
    rebalance,
    check_exit,
    exit_position,
)
from src.funding.screen import FundingCandidate, FundingScreener
from src.universe import Universe
from src.okx.pool import OkxClientPool
from src.okx.trading import OkxTrading


class FundingPipeline:
    """Orchestrates the full funding capture lifecycle:
    screen -> enter -> manage -> exit.
    """

    def __init__(
        self,
        pool: OkxClientPool,
        trading: OkxTrading,
        config: FundingConfig,
        clickhouse: ClickHouseClient | None,
        logger: AppLogger,
    ):
        self._pool = pool
        self._trading = trading
        self._config = config
        self._ch = clickhouse
        self._logger = logger
        self._positions: list[FundingPosition] = []

    def run(self, universe: Universe) -> None:
        session_id = uuid.uuid4().hex
        self._logger.info(f"FundingPipeline session {session_id}")
        self._logger.info(f"Universe: {len(universe)} pairs")

        self._write_session(session_id, "RUNNING")

        try:
            screener = FundingScreener(self._pool, self._config, self._logger)
            candidates = screener.screen(universe)
            self._logger.info(f"Candidates: {len(candidates)}")
            self._write_screen(session_id, candidates)

            self._phase_exit()

            self._phase_enter(candidates)

            self._phase_manage()

            self._write_session(session_id, "SUCCESS")
            self._print_summary()

        except Exception as exc:
            self._write_session(session_id, "FAILED", error=str(exc))
            self._logger.error(f"FundingPipeline FAILED: {exc}", exc_info=True)
            raise

    def _phase_enter(self, candidates: list[FundingCandidate]) -> None:
        active_pairs = {p.pair for p in self._positions}
        for c in candidates:
            if c.pair in active_pairs:
                self._logger.info(f"enter: {c.pair} already active — skip")
                continue
            pos = enter_position(
                self._pool, self._trading, c.pair, c.direction,
                self._config, self._logger,
            )
            if pos is not None:
                self._positions.append(pos)
                self._logger.info(
                    f"enter: opened {c.pair} direction={c.direction} "
                    f"notional={pos.notional:.2f}"
                )

    def _phase_manage(self) -> None:
        for i, pos in enumerate(self._positions):
            self._positions[i] = rebalance(
                self._pool, self._trading, pos, self._config, self._logger,
            )

    def _phase_exit(self) -> None:
        remaining: list[FundingPosition] = []
        for pos in self._positions:
            should_exit = check_exit(self._pool, pos, self._logger)
            if should_exit:
                exit_position(self._trading, pos, self._logger)
            else:
                remaining.append(pos)
        self._positions = remaining

    def _print_summary(self) -> None:
        self._logger.info(f"Active positions: {len(self._positions)}")
        for pos in self._positions:
            self._logger.info(
                f"  {pos.pair} dir={pos.direction} notional={pos.notional:.2f} "
                f"spot_qty={pos.spot_qty:.6f} perp_qty={pos.perp_qty:.6f}"
            )

    def _write_session(self, session_id: str, status: str,
                       error: str | None = None) -> None:
        if self._ch is None:
            return
        self._ch.write("funding_session", {
            "session_id": session_id,
            "timestamp": _now_ms(),
            "status": status,
            "error_message": error,
        })

    def _write_screen(self, session_id: str,
                      candidates: list[FundingCandidate]) -> None:
        if self._ch is None:
            return
        for c in candidates:
            self._ch.write("funding_screen", {
                "session_id": session_id,
                "pair": c.pair,
                "inst_id": c.inst_id,
                "direction": c.direction,
                "funding_rate": c.funding_rate,
                "timestamp": _now_ms(),
            })


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
