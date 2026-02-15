from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def periods_per_year(steps: str) -> int:
    """Convert timeframe string to number of periods per year."""
    multipliers = {
        "m": 525600,   # minutes per year
        "h": 8760,     # hours per year
        "d": 365,      # days per year
        "w": 52,       # weeks per year
        "M": 12,       # months per year
    }
    if not steps:
        return 525600
    unit = steps[-1]
    try:
        count = int(steps[:-1]) if len(steps) > 1 else 1
    except ValueError:
        return 525600
    base = multipliers.get(unit, 525600)
    return base // count if count > 0 else base


@dataclass
class InfluxValue:
    enabled: bool = False
    url: str = "http://localhost:8086"
    org: str = "trading"
    bucket: str = "trading"
    token: str = ""
    write_behavior: dict[str, Any] = field(default_factory=dict)


@dataclass
class OkxValue:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    demo: bool = True


@dataclass
class CrossMAValue:
    short_length: int = 15
    long_length: int = 200
    equity: Any = "auto"
    sizing: dict[str, Any] = field(default_factory=lambda: {"default": 1.0})


@dataclass
class DrawdownValue:
    window: int = 500
    threshold_scale_map: dict[str, Any] = field(default_factory=lambda: {"0": 1.0})
    risk_limits: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeValue:
    mode: str = "live"
    instrument: str = ""
    steps: str = "1m"


@dataclass
class BacktestValue:
    start: str = ""
    end: str = ""


@dataclass
class ConfigValue:
    log_level: str = "INFO"
    influx: InfluxValue = field(default_factory=InfluxValue)
    okx: OkxValue = field(default_factory=OkxValue)
    crossma: CrossMAValue = field(default_factory=CrossMAValue)
    drawdown: DrawdownValue = field(default_factory=DrawdownValue)
    trade: TradeValue = field(default_factory=TradeValue)
    backtest: BacktestValue = field(default_factory=BacktestValue)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigValue":
        influx = _as_dict(data.get("influx"))
        okx = _as_dict(data.get("okx"))
        crossma = _as_dict(data.get("crossma"))
        drawdown = _as_dict(data.get("drawdown"))
        trade = _as_dict(data.get("trade"))
        backtest = _as_dict(data.get("backtest"))

        return cls(
            log_level=str(data.get("log_level", "INFO")),
            influx=InfluxValue(
                enabled=bool(influx.get("enabled", False)),
                url=str(influx.get("url", "http://localhost:8086")),
                org=str(influx.get("org", "trading")),
                bucket=str(influx.get("bucket", "trading")),
                token=str(influx.get("token", "")),
                write_behavior=_as_dict(influx.get("write_behavior")),
            ),
            okx=OkxValue(
                api_key=str(okx.get("api_key", "")),
                secret_key=str(okx.get("secret_key", "")),
                passphrase=str(okx.get("passphrase", "")),
                demo=bool(okx.get("demo", True)),
            ),
            crossma=CrossMAValue(
                short_length=int(crossma.get("short_length", 15)),
                long_length=int(crossma.get("long_length", 200)),
                equity=crossma.get("equity", "auto"),
                sizing=_as_dict(crossma.get("sizing")) or {"default": 1.0},
            ),
            drawdown=DrawdownValue(
                window=int(drawdown.get("window", 500)),
                threshold_scale_map=_as_dict(drawdown.get("threshold_scale_map")) or {"0": 1.0},
                risk_limits=_as_dict(drawdown.get("risk_limits")),
            ),
            trade=TradeValue(
                mode=str(trade.get("mode", "live")),
                instrument=str(trade.get("instrument", "")),
                steps=str(trade.get("steps", "1m")),
            ),
            backtest=BacktestValue(
                start=str(backtest.get("start", "")),
                end=str(backtest.get("end", "")),
            ),
        )


@dataclass
class AppConfig:
    value: ConfigValue
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> ConfigValue:
        return self.value

    @classmethod
    def load_yaml(cls, config_path: str | None = None) -> "AppConfig":
        import yaml

        path = Path(config_path or "setup.yaml")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        return cls(value=ConfigValue.from_dict(data), raw=data)
