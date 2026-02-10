"""MACD (Moving Average Convergence Divergence) strategy."""

import pandas as pd


class Strategy:
    """MACD crossover strategy.

    Rules
    -----
    - When the MACD line crosses **above** the signal line → go **long** (+1).
    - When the MACD line crosses **below** the signal line → go **short** (−1).
    - Position is held until the next crossover.

    Parameters
    ----------
    fast : int | str
        Period of the fast EMA (default 12).
    slow : int | str
        Period of the slow EMA (default 26).
    signal : int | str
        Period of the signal EMA applied to the MACD line (default 9).
    source : str
        Column name to compute EMAs on (required).
    """

    def __init__(
        self,
        fast: int | str = 12,
        slow: int | str = 26,
        signal: int | str = 9,
        *,
        source: str,
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self.source = source

        if self.fast >= self.slow:
            raise ValueError(
                f"fast period ({self.fast}) must be less than slow period ({self.slow})"
            )

    def __repr__(self) -> str:
        return (
            f"MACD(fast={self.fast}, slow={self.slow}, "
            f"signal={self.signal}, source={self.source})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD, signal line, histogram, and position signals.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (must contain the column specified by *source*).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:

            - ``macd`` – MACD line (fast EMA − slow EMA)
            - ``macd_signal`` – signal line (EMA of MACD)
            - ``macd_hist`` – histogram (MACD − signal)
            - ``signal`` – raw crossover signal (+1 long, −1 short, 0 no change)
            - ``position`` – held position at each bar (+1 or −1)
        """
        if self.source not in df.columns:
            raise ValueError(f"Missing required column: '{self.source}'")

        out = df.copy()

        ema_fast = out[self.source].ewm(span=self.fast, adjust=False).mean()
        ema_slow = out[self.source].ewm(span=self.slow, adjust=False).mean()

        out["macd"] = ema_fast - ema_slow
        out["macd_signal"] = out["macd"].ewm(span=self.signal, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        # +1 when MACD > signal, -1 when MACD < signal
        out["signal"] = 0
        out.loc[out["macd"] > out["macd_signal"], "signal"] = 1
        out.loc[out["macd"] < out["macd_signal"], "signal"] = -1

        # Position changes only on crossovers; forward-fill to hold between them
        out["position"] = out["signal"]
        out["position"] = out["position"].replace(0, pd.NA).ffill().fillna(0).astype(int)

        return out
