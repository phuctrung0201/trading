from collections.abc import Iterator

from src.app.logger import AppLogger
from src.exchange.dto import MarketTrade
from src.okx.pool import OkxClientPool


class IngestPoller:
    """Fetches historical trades in 1-hour chunks using a shared OKX client pool.

    Each hourly chunk is dispatched to a thread via the pool's map method,
    bounded by pool_size concurrent workers.  All threads share the same
    rate-limited OkxClientPool.
    """

    ONE_HOUR_MS = 3_600_000

    def __init__(self, pool: OkxClientPool, instrument: str, logger: AppLogger):
        self._pool = pool
        self._instrument = instrument
        self._logger = logger

    def poll(self, start_ms: int, end_ms: int) -> Iterator[MarketTrade]:
        range_specs = self._build_hourly_specs(start_ms, end_ms)
        workers = min(self._pool.pool_size, len(range_specs))
        self._logger.info(
            f"IngestPoller workers={workers} chunks={len(range_specs)}"
        )

        all_trades: list[dict] = []
        chunk_results = self._pool.map(
            self._fetch_chunk, range_specs, max_workers=workers,
        )
        for i, chunk_trades in enumerate(chunk_results):
            self._logger.info(f"IngestPoller chunk={i} fetched={len(chunk_trades)}")
            all_trades.extend(chunk_trades)

        unique = self._dedup(all_trades)
        self._logger.info(
            f"IngestPoller raw={len(all_trades)} unique={len(unique)}"
        )
        unique.sort(key=lambda t: int(t["ts"]))
        for t in unique:
            if int(t["ts"]) >= start_ms:
                yield MarketTrade(
                    trade_id=t["tradeId"],
                    timestamp=t["ts"],
                    price=float(t["px"]),
                    size=float(t["sz"]),
                    side=t["side"],
                )

    def _build_hourly_specs(
        self, start_ms: int, end_ms: int,
    ) -> list[tuple[int, int, str, int]]:
        specs: list[tuple[int, int, str, int]] = []
        cursor = start_ms
        idx = 0
        while cursor < end_ms:
            r_end = min(cursor + self.ONE_HOUR_MS, end_ms)
            specs.append((cursor, r_end, self._instrument, idx))
            cursor = r_end
            idx += 1
        return specs

    def _fetch_chunk(
        self,
        pool: OkxClientPool,
        range_spec: tuple[int, int, str, int],
    ) -> list[dict]:
        range_start_ms, range_end_ms, inst_id, chunk_idx = range_spec
        trades: list[dict] = []
        cursor = str(range_end_ms)
        batch = 0
        while True:
            raw = pool.public_get(
                "/api/v5/market/history-trades",
                params={
                    "instId": inst_id,
                    "limit": "100",
                    "after": cursor,
                    "type": "2",
                },
            )
            if not raw:
                break
            batch += 1
            trades.extend(raw)
            oldest_ts = int(raw[-1]["ts"])
            self._logger.info(
                f"IngestPoller chunk={chunk_idx} batch={batch} got={len(raw)} "
                f"range=[{range_start_ms}..{range_end_ms}] "
                f"oldest_ts={oldest_ts} total={len(trades)}"
            )
            if oldest_ts <= range_start_ms or len(raw) < 100:
                break
            if str(oldest_ts) == cursor:
                break
            cursor = str(oldest_ts)
        return trades

    @staticmethod
    def _dedup(trades: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for t in trades:
            tid = t["tradeId"]
            if tid not in seen:
                seen.add(tid)
                unique.append(t)
        return unique
