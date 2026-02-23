from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.app.config import PortfolioConfig
from src.app.logger import AppLogger
from src.clickhouse.client import ClickHouseClient
from src.okx.pool import OkxClientPool
from src.portfolio.rank import RankResult, rank
from src.portfolio.screen import ScreenResult, Screener
from src.portfolio.universe import UniverseFetcher


class Pipeline:
    def __init__(self, pool: OkxClientPool, config: PortfolioConfig,
                 clickhouse: ClickHouseClient | None, logger: AppLogger):
        self._pool = pool
        self._config = config
        self._ch = clickhouse
        self._logger = logger

    def run(self) -> list[RankResult]:
        session_id = uuid.uuid4().hex
        self._logger.info(f"Pipeline session={session_id}")
        started_at = _now_ms()

        self._write_session(session_id, started_at, "RUNNING")

        try:
            instruments = UniverseFetcher(
                self._pool, self._config.universe,
                self._config.screening, self._logger,
            ).fetch()

            screener = Screener(
                self._config.screening, self._config.ranking, self._logger,
            )
            screen_results = screener.screen(instruments)
            self._write_screen(session_id, screen_results)

            passed = [s for s in screen_results if s.passed]
            rankings = rank(passed, self._config.ranking)
            self._write_ranking(session_id, rankings)

            self._write_session(
                session_id, started_at, "SUCCESS",
                universe_size=len(screen_results),
                passed_count=len(rankings),
            )

            self._print_summary(screen_results, rankings)
            self._logger.info(f"Pipeline session={session_id} SUCCESS")
            return rankings

        except Exception as exc:
            self._write_session(
                session_id, started_at, "FAILED", error=str(exc),
            )
            self._logger.error(f"Pipeline session={session_id} FAILED: {exc}")
            raise

        finally:
            if self._ch is not None:
                self._ch.close()

    def _print_summary(self, screen: list[ScreenResult],
                       rankings: list[RankResult]):
        total = len(screen)
        passed = len(rankings)
        failed = total - passed
        self._logger.info(f"Universe={total}  Passed={passed}  Failed={failed}")
        if rankings:
            self._logger.info("--- Rankings ---")
            for r in rankings:
                self._logger.info(
                    f"  #{r.rank} {r.instrument}  "
                    f"score={r.composite_score:.4f}  "
                    f"adf_p={r.adf_pvalue:.4f}  "
                    f"hurst={r.hurst:.4f}  "
                    f"hl={r.half_life:.1f}  "
                    f"vol={r.volatility:.4f}"
                )

    # -- ClickHouse writes ---------------------------------------------------

    def _write_session(self, session_id: str, started_at: int, status: str,
                       universe_size: int = 0, passed_count: int = 0,
                       error: str | None = None):
        if self._ch is None:
            return
        self._ch.write("portfolio_session", {
            "session_id": session_id,
            "started_at": started_at,
            "finished_at": _now_ms() if status != "RUNNING" else None,
            "status": status,
            "config_name": "scanner",
            "universe_size": universe_size,
            "passed_count": passed_count,
            "error_message": error,
        })

    def _write_screen(self, session_id: str, results: list[ScreenResult]):
        if self._ch is None:
            return
        for s in results:
            self._ch.write("portfolio_screen", {
                "session_id": session_id,
                "instrument": s.instrument,
                "adf_stat": s.adf_stat,
                "adf_pvalue": s.adf_pvalue,
                "hurst": s.hurst,
                "half_life": s.half_life,
                "volatility": s.volatility,
                "daily_volume": s.daily_volume,
                "passed": 1 if s.passed else 0,
                "fail_reason": s.fail_reason,
            })

    def _write_ranking(self, session_id: str, rankings: list[RankResult]):
        if self._ch is None:
            return
        for r in rankings:
            self._ch.write("portfolio_ranking", {
                "session_id": session_id,
                "instrument": r.instrument,
                "rank": r.rank,
                "composite_score": r.composite_score,
                "adf_pvalue": r.adf_pvalue,
                "hurst": r.hurst,
                "half_life": r.half_life,
                "volatility": r.volatility,
            })


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
