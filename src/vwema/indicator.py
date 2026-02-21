import math
from collections import deque


class VWEMA:
    """Volume-Weighted Exponential Moving Average.

    Each price update is weighted by normalized volume so that
    high-volume trades pull the average harder than low-volume ones.

    Volume is normalized against a rolling median so the effective
    smoothing stays stable across instruments and over time.

    When volume=1.0 (the default) or normalization is disabled,
    this reduces to a standard EMA.
    """

    def __init__(self, period: int, normalize_window: int = 200):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._value: float | None = None
        self._normalize = normalize_window > 0
        self._vol_buffer: deque[float] = deque(maxlen=normalize_window if self._normalize else 0)
        self._sorted_vols: list[float] = []

    def _rolling_median(self) -> float:
        n = len(self._sorted_vols)
        if n == 0:
            return 1.0
        mid = n // 2
        if n % 2 == 1:
            return self._sorted_vols[mid]
        return (self._sorted_vols[mid - 1] + self._sorted_vols[mid]) / 2.0

    def _add_volume(self, volume: float) -> None:
        if self._vol_buffer.maxlen == 0:
            return
        evicted = None
        if len(self._vol_buffer) == self._vol_buffer.maxlen:
            evicted = self._vol_buffer[0]
        self._vol_buffer.append(volume)
        self._insert_sorted(volume)
        if evicted is not None:
            self._remove_sorted(evicted)

    def _insert_sorted(self, v: float) -> None:
        lo, hi = 0, len(self._sorted_vols)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted_vols[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        self._sorted_vols.insert(lo, v)

    def _remove_sorted(self, v: float) -> None:
        lo, hi = 0, len(self._sorted_vols)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted_vols[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self._sorted_vols) and self._sorted_vols[lo] == v:
            self._sorted_vols.pop(lo)

    def _normalize_volume(self, volume: float) -> float:
        if not self._normalize:
            return volume
        self._add_volume(volume)
        median = self._rolling_median()
        if median <= 0:
            return 1.0
        return volume / median

    def update(self, price: float, volume: float = 1.0) -> float:
        """Feed a new price tick and return the updated VWEMA.

        Treats a trade of normalized volume v as v "ticks" at that price:
            effective_alpha = 1 - (1 - alpha)^v
        Large trades move the average more; tiny trades barely nudge it.
        """
        norm_vol = self._normalize_volume(volume)
        if self._value is None:
            self._value = price
            return self._value
        w = 1.0 - math.pow(1.0 - self.alpha, norm_vol)
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
        self._vol_buffer.clear()
        self._sorted_vols.clear()
