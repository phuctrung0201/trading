import math


class VWEMA:
    """Volume-Weighted Exponential Moving Average.

    Each price update is weighted by volume so that high-volume trades
    pull the average harder than low-volume ones.

    When volume=1.0 (the default) this reduces to a standard EMA.
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._value: float | None = None

    def update(self, price: float, volume: float = 1.0) -> float:
        """Feed a new price tick and return the updated VWEMA.

        Treats a trade of volume v as v "ticks" at that price:
            effective_alpha = 1 - (1 - alpha)^v
        Large trades move the average more; tiny trades barely nudge it.
        """
        if self._value is None:
            self._value = price
            return self._value
        if volume == 1.0:
            w = self.alpha
        else:
            w = 1.0 - math.pow(1.0 - self.alpha, volume)
        self._value = w * price + (1.0 - w) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None

    def reset(self) -> None:
        self._value = None
