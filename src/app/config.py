from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_type_hints, get_origin, get_args

import yaml

CONFIG_DIR = Path("config")


class Config:
    """Base class for all config schemas."""

    @classmethod
    def from_dict(cls, data: dict):
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            raw = data.get(f.name)
            if raw is None:
                continue
            expected = _unwrap_optional(hints[f.name])
            if isinstance(raw, dict) and _is_config_type(expected):
                kwargs[f.name] = expected.from_dict(raw)
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)


def _is_config_type(cls) -> bool:
    return isinstance(cls, type) and issubclass(cls, Config)


def _unwrap_optional(tp):
    if get_origin(tp) is type(int | None):
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def load_config[T: Config](module: str, name: str, schema: type[T]) -> T:
    """Load config/<module>/<name>.yaml and return as typed config."""
    path = CONFIG_DIR / module / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return schema.from_dict(data)


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------

@dataclass
class LoggerConfig(Config):
    log_level: str = "INFO"


@dataclass
class OkxConfig(Config):
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    demo: bool = True


@dataclass
class ClickHouseConfig(Config):
    enabled: bool = False
    url: str = "http://localhost:8123"
    database: str = "trading"
    user: str = "default"
    password: str = ""


@dataclass
class DepthConfig(Config):
    hour: int = 0
    minute: int = 0

    @property
    def total_minutes(self) -> int:
        return self.hour * 60 + self.minute


@dataclass
class CrossMAConfig(Config):
    short_length: int = 15
    long_length: int = 200
    bucket_interval: str = "5m"


@dataclass
class DrawdownConfig(Config):
    window: int = 500
    threshold_scale_map: dict[float, float] = field(
        default_factory=lambda: {0.0: 1.0},
    )


@dataclass
class BacktestConfig(Config):
    dataset: str | None = None
    depth: DepthConfig | None = None
    fee_rate: float = 0.0
    initial_equity: float = 100.0


@dataclass
class MeanRevConfig(Config):
    lookback: int
    entry_threshold: float
    exit_threshold: float
    bucket_interval: str = "5m"


@dataclass
class UniverseConfig(Config):
    quote: str = "USDT"
    type: str = "SWAP"
    min_24h_volume_usd: float = 10_000_000


@dataclass
class ScreeningConfig(Config):
    bucket_interval: str = "5m"
    lookback_hours: int = 6
    sma_length: int = 72
    adf_pvalue_max: float = 0.05
    hurst_max: float = 0.5
    half_life_min: float = 10
    half_life_max: float = 5000


@dataclass
class RankingWeightsConfig(Config):
    adf_score: float = 0.4
    hurst_score: float = 0.4
    half_life_score: float = 0.2


@dataclass
class RankingConfig(Config):
    adf_pvalue_max: float = 0.05
    hurst_max: float = 0.5
    half_life_min: float = 10
    half_life_max: float = 5000
    top_n: int = 10
    weights: RankingWeightsConfig = field(default_factory=RankingWeightsConfig)


@dataclass
class PortfolioConfig(Config):
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)


@dataclass
class SetupConfig(Config):
    instrument: str = ""
    leverage: int = 1
    strategy: str = "drawdown_crossma"
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    crossma: CrossMAConfig = field(default_factory=CrossMAConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    meanrev: MeanRevConfig = field(default_factory=lambda: MeanRevConfig(
        lookback=100, entry_threshold=2.0, exit_threshold=0.5,
    ))
