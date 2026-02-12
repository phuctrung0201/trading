"""Moving-average crossover (MACROSS) strategy using EMA.

Uses pandas_ta for EMA calculation, which matches TradingView / OKX charts
(SMA-seeded, α = 2 / (period + 1)).
"""

import pandas as pd
import pandas_ta as ta


class MACross:
    """EMA-based moving-average crossover strategy.

    Rules
    -----
    - When the short EMA crosses **above** the long EMA → go **long** (+1).
    - When the short EMA crosses **below** the long EMA → go **short** (−1).
    - Position is held until the next crossover.

    The EMA is computed using the TradingView / OKX method: seeded with
    the SMA of the first *period* bars, then applying the standard
    recursive formula with α = 2 / (period + 1).

    Parameters
    ----------
    short : int | str
        Period of the short (faster) EMA.
    long : int | str
        Period of the long (slower) EMA.
    source : str
        Column name to compute EMAs on.
        Can be any OHLCV column, e.g. ``"close"``, ``"open"``, ``"high"``.
        Defaults to ``"close"``.
    """

    def __init__(self, short: int | str = 10, long: int | str = 20, source: str = "close"):
        self.short = int(short)
        self.long = int(long)
        self.source = source

        if self.short >= self.long:
            raise ValueError(
                f"short period ({self.short}) must be less than long period ({self.long})"
            )

    def __repr__(self) -> str:
        return f"MACross(short={self.short}, long={self.long}, source={self.source})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA crossover and position signals.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (must contain the column specified by *source*).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:

            - ``ema_short`` – short exponential moving average
            - ``ema_long`` – long exponential moving average
            - ``signal`` – raw crossover signal (+1 long, −1 short, 0 no change)
            - ``position`` – held position at each bar (+1 or −1)
        """
        if self.source not in df.columns:
            raise ValueError(f"Missing required column: '{self.source}'")

        out = df.copy()
        out["ema_short"] = ta.ema(out[self.source], length=self.short)
        out["ema_long"] = ta.ema(out[self.source], length=self.long)

        # Detect crossover points only (NaN-safe)
        diff = out["ema_short"] - out["ema_long"]
        cross_up = (diff > 0) & (diff.shift(1) <= 0)   # short crosses above long
        cross_down = (diff < 0) & (diff.shift(1) >= 0)  # short crosses below long

        out["signal"] = 0
        out.loc[cross_up, "signal"] = 1
        out.loc[cross_down, "signal"] = -1

        # Hold position between crossovers
        out["position"] = out["signal"]
        out["position"] = out["position"].replace(0, pd.NA).ffill().fillna(0).astype(int)

        return out
