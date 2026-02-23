from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller

from src.app.config import ScreeningConfig
from src.app.logger import AppLogger
from src.portfolio.universe import Instrument, _parse_bucket_interval


@dataclass
class ScreenResult:
    instrument: str
    daily_volume: float
    adf_stat: float | None
    adf_pvalue: float | None
    hurst: float | None
    half_life: float | None
    volatility: float | None
    passed: bool
    fail_reason: str | None


def _bucket_prices(trades: list[dict], interval_sec: int) -> list[float]:
    """Group trades into time buckets; pick the price with the largest
    total volume in each bucket."""
    if not trades:
        return []

    start_ms = int(trades[0]["ts"])
    interval_ms = interval_sec * 1000

    buckets: dict[int, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    for t in trades:
        ts = int(t["ts"])
        idx = (ts - start_ms) // interval_ms
        price = float(t["px"])
        size = float(t["sz"])
        buckets[idx][price] += size

    if not buckets:
        return []

    max_idx = max(buckets.keys())
    prices: list[float] = []
    for i in range(max_idx + 1):
        if i in buckets:
            best_price = max(buckets[i].items(), key=lambda kv: kv[1])[0]
            prices.append(best_price)
        elif prices:
            prices.append(prices[-1])

    return prices


def _compute_adf(prices: np.ndarray) -> tuple[float, float]:
    result = adfuller(prices, autolag="AIC")
    return float(result[0]), float(result[1])


def _compute_hurst(prices: np.ndarray) -> float:
    """Rescaled range (R/S) Hurst exponent."""
    n = len(prices)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 128)
    sizes = []
    k = 8
    while k <= max_k:
        sizes.append(k)
        k = int(k * 1.5)
    if not sizes:
        return 0.5

    rs_means = []
    for size in sizes:
        n_chunks = n // size
        if n_chunks == 0:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = prices[i * size:(i + 1) * size]
            mean = np.mean(chunk)
            deviations = chunk - mean
            cumdev = np.cumsum(deviations)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(chunk, ddof=1)
            if s > 1e-12:
                rs_values.append(r / s)
        if rs_values:
            rs_means.append((size, np.mean(rs_values)))

    if len(rs_means) < 2:
        return 0.5

    log_n = np.array([math.log(s) for s, _ in rs_means])
    log_rs = np.array([math.log(rs) for _, rs in rs_means])

    coeffs = np.polyfit(log_n, log_rs, 1)
    return float(coeffs[0])


def _compute_half_life(prices: np.ndarray) -> float | None:
    """OLS half-life: Δp(t) = λ·p(t-1) + ε."""
    lag = prices[:-1]
    delta = np.diff(prices)

    if len(lag) < 2:
        return None

    lag_mean = np.mean(lag)
    lag_centered = lag - lag_mean
    denom = np.dot(lag_centered, lag_centered)
    if denom < 1e-15:
        return None

    lam = np.dot(lag_centered, delta) / denom
    if lam >= 0:
        return None

    return float(-math.log(2) / lam)


def _compute_volatility(prices: np.ndarray, interval_sec: int) -> float:
    returns = np.diff(prices) / prices[:-1]
    buckets_per_year = (365 * 24 * 3600) / interval_sec
    return float(np.std(returns, ddof=1) * math.sqrt(buckets_per_year))


class Screener:
    def __init__(self, screening_cfg: ScreeningConfig, logger: AppLogger):
        self._cfg = screening_cfg
        self._logger = logger
        self._interval_sec = _parse_bucket_interval(screening_cfg.bucket_interval)

    def screen(self, instruments: list[Instrument]) -> list[ScreenResult]:
        results: list[ScreenResult] = []
        for i, inst in enumerate(instruments):
            self._logger.info(
                f"Screening [{i+1}/{len(instruments)}] {inst.inst_id}"
            )
            result = self._screen_one(inst)
            results.append(result)
            status = "PASS" if result.passed else f"FAIL ({result.fail_reason})"
            self._logger.info(f"  {inst.inst_id}: {status}")
        return results

    def _screen_one(self, inst: Instrument) -> ScreenResult:
        prices = _bucket_prices(inst.trades, self._interval_sec)

        min_points = self._cfg.sma_length + 10
        if len(prices) < min_points:
            return ScreenResult(
                instrument=inst.inst_id,
                daily_volume=inst.daily_volume_usd,
                adf_stat=None, adf_pvalue=None, hurst=None,
                half_life=None, volatility=None,
                passed=False,
                fail_reason=f"insufficient data ({len(prices)} buckets)",
            )

        arr = np.array(prices, dtype=np.float64)
        failures: list[str] = []

        try:
            adf_stat, adf_pvalue = _compute_adf(arr)
        except Exception as exc:
            adf_stat, adf_pvalue = None, None
            failures.append("adf_error")
            self._logger.warning(
                f"_compute_adf failed instrument={inst.inst_id} "
                f"data_points={len(arr)}: {exc}",
                exc_info=True,
            )

        try:
            hurst = _compute_hurst(arr)
        except Exception as exc:
            hurst = None
            failures.append("hurst_error")
            self._logger.warning(
                f"_compute_hurst failed instrument={inst.inst_id} "
                f"data_points={len(arr)}: {exc}",
                exc_info=True,
            )

        try:
            half_life = _compute_half_life(arr)
        except Exception as exc:
            half_life = None
            failures.append("half_life_error")
            self._logger.warning(
                f"_compute_half_life failed instrument={inst.inst_id} "
                f"data_points={len(arr)}: {exc}",
                exc_info=True,
            )

        try:
            volatility = _compute_volatility(arr, self._interval_sec)
        except Exception as exc:
            volatility = None
            self._logger.warning(
                f"_compute_volatility failed instrument={inst.inst_id} "
                f"data_points={len(arr)} interval_sec={self._interval_sec}: {exc}",
                exc_info=True,
            )

        if adf_pvalue is not None and adf_pvalue > self._cfg.adf_pvalue_max:
            failures.append(f"adf_pvalue={adf_pvalue:.4f}")
        if hurst is not None and hurst > self._cfg.hurst_max:
            failures.append(f"hurst={hurst:.4f}")
        if half_life is None:
            if "half_life_error" not in failures:
                failures.append("half_life=non_reverting")
        elif half_life < self._cfg.half_life_min:
            failures.append(f"half_life={half_life:.1f}<min")
        elif half_life > self._cfg.half_life_max:
            failures.append(f"half_life={half_life:.1f}>max")

        passed = len(failures) == 0
        fail_reason = "; ".join(failures) if failures else None

        return ScreenResult(
            instrument=inst.inst_id,
            daily_volume=inst.daily_volume_usd,
            adf_stat=adf_stat,
            adf_pvalue=adf_pvalue,
            hurst=hurst,
            half_life=half_life,
            volatility=volatility,
            passed=passed,
            fail_reason=fail_reason,
        )
