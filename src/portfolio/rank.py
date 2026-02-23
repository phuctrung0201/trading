from __future__ import annotations

from dataclasses import dataclass

from src.app.config import RankingConfig
from src.portfolio.screen import ScreenResult


@dataclass
class RankResult:
    instrument: str
    rank: int
    composite_score: float
    adf_pvalue: float
    hurst: float
    half_life: float
    volatility: float


def rank(passed: list[ScreenResult], cfg: RankingConfig) -> list[RankResult]:
    if not passed:
        return []

    w = cfg.weights
    hl_min = cfg.half_life_min
    hl_max = cfg.half_life_max
    hl_mid = (hl_min + hl_max) / 2
    hl_range = (hl_max - hl_min) / 2

    scored: list[tuple[float, ScreenResult]] = []
    for s in passed:
        adf_norm = 1.0 - (s.adf_pvalue / cfg.adf_pvalue_max)
        hurst_norm = 1.0 - (s.hurst / cfg.hurst_max)
        hl_norm = 1.0 - abs(s.half_life - hl_mid) / hl_range if hl_range > 0 else 0.5

        adf_norm = max(0.0, min(1.0, adf_norm))
        hurst_norm = max(0.0, min(1.0, hurst_norm))
        hl_norm = max(0.0, min(1.0, hl_norm))

        score = (
            w.adf_score * adf_norm
            + w.hurst_score * hurst_norm
            + w.half_life_score * hl_norm
        )
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[RankResult] = []
    for i, (score, s) in enumerate(scored):
        results.append(RankResult(
            instrument=s.instrument,
            rank=i + 1,
            composite_score=round(score, 6),
            adf_pvalue=s.adf_pvalue,
            hurst=s.hurst,
            half_life=s.half_life,
            volatility=s.volatility or 0.0,
        ))
    return results
