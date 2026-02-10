"""Dual moving-average crossover strategy."""

import pandas as pd


class Strategy:
    """Dual moving-average crossover strategy.

    Rules
    -----
    - When the fast MA crosses **above** the slow MA → go **long** (+1).
    - When the fast MA crosses **below** the slow MA → go **short** (−1).
    - Position is held until the next crossover.

    Parameters
    ----------
    fast : int | str
        Period of the fast (shorter) moving average.
    slow : int | str
        Period of the slow (longer) moving average.
    source : str
        Column name to compute moving averages on (default ``"close"``).
        Can be any OHLCV column, e.g. ``"open"``, ``"high"``, ``"low"``.
    """

    def __init__(self, fast: int | str = 10, slow: int | str = 20, *, source: str):
        self.fast = int(fast)
        self.slow = int(slow)
        self.source = source

        if self.fast >= self.slow:
            raise ValueError(
                f"fast period ({self.fast}) must be less than slow period ({self.slow})"
            )

    def __repr__(self) -> str:
        return f"DoubleMA(fast={self.fast}, slow={self.slow}, source={self.source})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving averages and position signals.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (must contain the column specified by *source*).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:

            - ``ma_fast`` – fast moving average
            - ``ma_slow`` – slow moving average
            - ``signal`` – raw crossover signal (+1 long, −1 short, 0 no change)
            - ``position`` – held position at each bar (+1 or −1)
        """
        if self.source not in df.columns:
            raise ValueError(f"Missing required column: '{self.source}'")

        out = df.copy()
        out["ma_fast"] = out[self.source].rolling(window=self.fast).mean()
        out["ma_slow"] = out[self.source].rolling(window=self.slow).mean()

        # +1 when fast > slow, -1 when fast < slow
        out["signal"] = 0
        out.loc[out["ma_fast"] > out["ma_slow"], "signal"] = 1
        out.loc[out["ma_fast"] < out["ma_slow"], "signal"] = -1

        # Position changes only on crossovers; forward-fill to hold between them
        out["position"] = out["signal"]
        out["position"] = out["position"].replace(0, pd.NA).ffill().fillna(0).astype(int)

        return out
