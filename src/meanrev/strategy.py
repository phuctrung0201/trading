from src.math.rolling import RollingWindow
from src.zscore.indicator import zscore


class MeanRevIndicator:
    def __init__(self, lookback: int, entry_threshold: float, exit_threshold: float):
        self._window = RollingWindow(lookback)
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._direction: str | None = None
        self._last_z: float | None = None

    def push(self, price: float) -> str | None:
        self._window.push(price)
        if not self._window.is_ready():
            self._last_z = None
            return None

        z = zscore(price, self._window.mean(), self._window.std())
        self._last_z = z
        if z is None:
            return None

        if self._direction is not None:
            if abs(z) < self._exit_threshold:
                self._direction = None
                return "EXIT"
            if z < -self._entry_threshold and self._direction != "LONG":
                self._direction = "LONG"
                return "LONG"
            if z > self._entry_threshold and self._direction != "SHORT":
                self._direction = "SHORT"
                return "SHORT"
            return None

        if z < -self._entry_threshold:
            self._direction = "LONG"
            return "LONG"
        if z > self._entry_threshold:
            self._direction = "SHORT"
            return "SHORT"
        return None

    def is_ready(self) -> bool:
        return self._window.is_ready()

    @property
    def last_z(self) -> float | None:
        return self._last_z
