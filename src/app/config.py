from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar, Type, get_type_hints, get_origin, get_args

import yaml

CONFIG_DIR = Path("config")
T = TypeVar("T")


def _is_dataclass_type(cls) -> bool:
    return hasattr(cls, "__dataclass_fields__")


def _unwrap_optional(tp):
    if get_origin(tp) is type(int | None):
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _unmarshal(cls: Type[T], data: dict) -> T:
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        raw = data.get(f.name)
        if raw is None:
            continue
        expected = _unwrap_optional(hints[f.name])
        if _is_dataclass_type(expected) and isinstance(raw, dict):
            kwargs[f.name] = _unmarshal(expected, raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


def load_config(module: str, name: str, schema: Type[T] | None = None) -> T | dict:
    """Load config/<module>/<name>.yaml and return as typed config or dict."""
    path = CONFIG_DIR / module / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    if schema is not None:
        return _unmarshal(schema, data)
    return data


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------

@dataclass
class LoggerConfig:
    log_level: str = "INFO"


@dataclass
class OkxConfig:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    demo: bool = True


@dataclass
class ClickHouseConfig:
    enabled: bool = False
    url: str = "http://localhost:8123"
    database: str = "trading"
    user: str = "default"
    password: str = ""


@dataclass
class DepthConfig:
    hour: int = 0
    minute: int = 0

    @property
    def total_minutes(self) -> int:
        return self.hour * 60 + self.minute


@dataclass
class CrossMAConfig:
    short_length: int = 15
    long_length: int = 200


@dataclass
class DrawdownConfig:
    window: int = 500
    threshold_scale_map: dict[float, float] = field(
        default_factory=lambda: {0.0: 1.0},
    )


@dataclass
class SetupConfig:
    instrument: str = ""
    leverage: int = 1
    strategy: str = "drawdown"
    depth: DepthConfig = field(default_factory=DepthConfig)
    crossma: CrossMAConfig = field(default_factory=CrossMAConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
