_STD_EPSILON = 1e-10


def zscore(value: float, mean: float, std: float) -> float | None:
    if std < _STD_EPSILON:
        return None
    return (value - mean) / std
