from pathlib import Path

import yaml

CONFIG_DIR = Path("config")


def load_config(module: str, name: str) -> dict:
    """Load config/<module>/<name>.yaml and return as dict."""
    path = CONFIG_DIR / module / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def periods_per_year(steps: str) -> int:
    """Convert timeframe string to number of periods per year."""
    multipliers = {
        "m": 525600,
        "h": 8760,
        "d": 365,
        "w": 52,
        "M": 12,
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
