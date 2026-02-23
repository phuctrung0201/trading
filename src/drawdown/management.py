from collections import deque


class DrawdownManagement:
    """Rolling-window drawdown tracking with tiered position scaling.

    Compose this into any strategy that needs drawdown-based sizing.
    Call `bootstrap_drawdown` once, then `calculate_drawdown` and
    `scale_position` on every bucket tick.
    """

    def __init__(self):
        self._equity_window: deque[float] = deque()
        self.position_sizing_map: dict[float, float] = {0.0: 1.0}
        self.drawdown_thresholds: list[float] = [0.0]

    def bootstrap_drawdown(self, window: int = 500,
                           threshold_scale_map: dict | None = None):
        self._equity_window = deque(maxlen=int(window))
        raw_map = threshold_scale_map or {0.0: 1.0}
        self.position_sizing_map = {float(k): float(v) for k, v in raw_map.items()}
        self.drawdown_thresholds = sorted(self.position_sizing_map.keys())

    def calculate_drawdown(self, equity: float) -> float:
        self._equity_window.append(equity)
        peak = max(self._equity_window)
        if peak <= 0:
            return 0.0
        return (equity - peak) / peak

    def scale_position(self, drawdown: float) -> float:
        dd = abs(drawdown)
        scale = float(self.position_sizing_map.get(0.0, 1.0))
        for threshold in self.drawdown_thresholds:
            if dd >= threshold:
                scale = float(self.position_sizing_map[threshold])
        return scale
