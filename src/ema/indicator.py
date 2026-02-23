class EMA:
    """Streaming Exponential Moving Average.

    Call ``update(price)`` once per bucket.  The bucket price should be
    the price with the largest total volume in that time window.
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._value: float | None = None

    def update(self, price: float) -> float:
        if self._value is None:
            self._value = price
        else:
            self._value = self.alpha * price + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None

    def reset(self) -> None:
        self._value = None
