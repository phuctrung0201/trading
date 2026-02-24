from __future__ import annotations

from dataclasses import dataclass

from src.app.config import FundingConfig
from src.app.logger import AppLogger
from src.funding.data import get_funding_rate, FundingRate
from src.okx.pool import OkxClientPool


@dataclass
class FundingCandidate:
    pair: str
    inst_id: str
    direction: str
    funding_rate: float


class FundingScreener:
    def __init__(self, pool: OkxClientPool, config: FundingConfig, logger: AppLogger):
        self._pool = pool
        self._config = config
        self._logger = logger

    def screen(self, universe: list[tuple[str, str]]) -> list[FundingCandidate]:
        """Screen a universe of (pair, perp_inst_id) and return candidates.

        A pair passes if |funding_rate| >= min_funding_rate.
        """
        candidates: list[FundingCandidate] = []
        for pair, perp_inst_id in universe:
            result = self._screen_one(pair, perp_inst_id)
            if result is not None:
                candidates.append(result)
        return candidates

    def _screen_one(self, pair: str, perp_inst_id: str) -> FundingCandidate | None:
        try:
            rate_data = get_funding_rate(self._pool, perp_inst_id)
        except Exception as exc:
            self._logger.warning(f"screen: failed to fetch rate for {perp_inst_id}: {exc}")
            return None

        rate = rate_data.rate
        if abs(rate) < self._config.min_funding_rate:
            self._logger.info(
                f"screen: {pair} rate={rate:.6f} < min={self._config.min_funding_rate:.6f} — skip"
            )
            return None

        direction = "long_spot" if rate > 0 else "short_spot"
        self._logger.info(f"screen: {pair} rate={rate:.6f} direction={direction} — candidate")

        return FundingCandidate(
            pair=pair,
            inst_id=perp_inst_id,
            direction=direction,
            funding_rate=rate,
        )
