from collections import deque
import math


class RollingWindow:
    def __init__(self, size: int):
        self._size = size
        self._buf: deque[float] = deque(maxlen=size)
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    def push(self, value: float):
        if len(self._buf) == self._size:
            old = self._buf[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._buf.append(value)
        self._sum += value
        self._sum_sq += value * value

    def mean(self) -> float:
        n = len(self._buf)
        if n == 0:
            return 0.0
        return self._sum / n

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        mean = self._sum / n
        variance = self._sum_sq / n - mean * mean
        if variance < 0:
            variance = 0.0
        return math.sqrt(variance)

    def is_ready(self) -> bool:
        return len(self._buf) == self._size
