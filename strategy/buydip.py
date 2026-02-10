"""Buy-the-Dip Mean Reversion strategy.

Enters long when price draws down significantly from its rolling high,
and enters short when price rallies significantly from its rolling low.
"""

import pandas as pd


class Strategy:
    """Mean-reversion strategy based on drawdown from rolling extremes.

    Rules
    -----
    - Compute the rolling high and rolling low over *lookback* bars.
    - When price drops **dip_pct %** below the rolling high → go **long** (+1).
    - When price rises **rally_pct %** above the rolling low → go **short** (−1).
    - Position is held until the opposite signal fires.

    Parameters
    ----------
    lookback : int
        Rolling window size for computing the high/low channel.
    dip_pct : float
        Drawdown percentage from the rolling high to trigger a long entry.
        E.g. ``1.5`` means buy when price is 1.5 % below the rolling high.
    rally_pct : float
        Rally percentage from the rolling low to trigger a short entry.
        E.g. ``1.5`` means sell when price is 1.5 % above the rolling low.
    """

    def __init__(
        self,
        lookback: int = 60,
        dip_pct: float = 1.5,
        rally_pct: float = 1.5,
    ):
        self.lookback = int(lookback)
        self.dip_pct = float(dip_pct)
        self.rally_pct = float(rally_pct)

    def __repr__(self) -> str:
        return (
            f"BuyDip(lookback={self.lookback}, "
            f"dip={self.dip_pct}%, rally={self.rally_pct}%)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute drawdown/rally signals and position.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (must contain a ``close`` column).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:

            - ``rolling_high`` – rolling high over *lookback* bars
            - ``rolling_low``  – rolling low over *lookback* bars
            - ``drawdown_pct`` – current drawdown from rolling high (negative %)
            - ``drawup_pct``   – current rally from rolling low (positive %)
            - ``signal``       – raw signal (+1 dip buy, −1 rally sell, 0 neutral)
            - ``position``     – held position at each bar
        """
        if "close" not in df.columns:
            raise ValueError("Missing required column: 'close'")

        out = df.copy()

        out["rolling_high"] = out["close"].rolling(self.lookback, min_periods=1).max()
        out["rolling_low"] = out["close"].rolling(self.lookback, min_periods=1).min()

        # Drawdown from rolling high (negative when below high)
        out["drawdown_pct"] = (
            (out["close"] - out["rolling_high"]) / out["rolling_high"] * 100
        )

        # Rally from rolling low (positive when above low)
        out["drawup_pct"] = (
            (out["close"] - out["rolling_low"]) / out["rolling_low"] * 100
        )

        # Generate signals
        out["signal"] = 0
        out.loc[out["drawdown_pct"] <= -self.dip_pct, "signal"] = 1   # buy the dip
        out.loc[out["drawup_pct"] >= self.rally_pct, "signal"] = -1   # sell the rally

        # Hold position until opposite signal
        out["position"] = out["signal"]
        out["position"] = (
            out["position"].replace(0, pd.NA).ffill().fillna(0).astype(int)
        )

        return out
