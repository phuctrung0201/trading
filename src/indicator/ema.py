class EMA:
    def calculate(self, values):
        if not values:
            return None

        period = len(values)
        alpha = 2.0 / (period + 1.0)
        ema = float(values[0])
        for value in values[1:]:
            ema = (float(value) * alpha) + (ema * (1.0 - alpha))
        return ema
