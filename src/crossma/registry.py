from src.drawdown.strategy import DrawdownCrossMAStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "crossma": DrawdownCrossMAStrategy,
    "drawdown": DrawdownCrossMAStrategy,
}


def get_strategy_class(name: str) -> type:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls


def register_strategy(name: str, cls: type):
    STRATEGY_REGISTRY[name] = cls
