"""Dip Recovery strategy – drawdown with bounce confirmation.

Instead of buying the dip blindly, this strategy waits for
price to actually start recovering before entering.

1. Price drops *dip_pct* % from the rolling high  → track the dip low.
2. Price bounces *recovery_pct* % off that low    → go **long**.
3. Price rallies *rally_pct* % from the rolling low → track the rally high.
4. Price pulls back *pullback_pct* % off that high  → go **short**.
5. Hold until the opposite signal fires.
"""

import numpy as np
import pandas as pd


class Strategy:
    """Drawdown recovery strategy with bounce/pullback confirmation.

    Parameters
    ----------
    lookback : int
        Rolling window for the high/low channel.
    dip_pct : float
        Drawdown % from rolling high to activate dip tracking.
    recovery_pct : float
        Bounce % from the dip low to confirm long entry.
    rally_pct : float
        Rally % from rolling low to activate rally tracking.
    pullback_pct : float
        Pullback % from the rally high to confirm short entry.
    """

    def __init__(
        self,
        lookback: int = 60,
        dip_pct: float = 2.0,
        recovery_pct: float = 0.5,
        rally_pct: float = 2.0,
        pullback_pct: float = 0.5,
    ):
        self.lookback = int(lookback)
        self.dip_pct = float(dip_pct)
        self.recovery_pct = float(recovery_pct)
        self.rally_pct = float(rally_pct)
        self.pullback_pct = float(pullback_pct)

    def __repr__(self) -> str:
        return (
            f"DipRecover(lookback={self.lookback}, "
            f"dip={self.dip_pct}%, recover={self.recovery_pct}%, "
            f"rally={self.rally_pct}%, pullback={self.pullback_pct}%)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute dip-recovery / rally-pullback signals.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (must contain a ``close`` column).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:

            - ``rolling_high``  – rolling high over *lookback* bars
            - ``rolling_low``   – rolling low over *lookback* bars
            - ``drawdown_pct``  – drawdown from rolling high (%)
            - ``drawup_pct``    – rally from rolling low (%)
            - ``position``      – held position (+1 long, −1 short, 0 flat)
        """
        if "close" not in df.columns:
            raise ValueError("Missing required column: 'close'")

        out = df.copy()

        out["rolling_high"] = out["close"].rolling(self.lookback, min_periods=1).max()
        out["rolling_low"] = out["close"].rolling(self.lookback, min_periods=1).min()

        out["drawdown_pct"] = (
            (out["close"] - out["rolling_high"]) / out["rolling_high"] * 100
        )
        out["drawup_pct"] = (
            (out["close"] - out["rolling_low"]) / out["rolling_low"] * 100
        )

        closes = out["close"].values
        dd = out["drawdown_pct"].values
        du = out["drawup_pct"].values
        positions = np.zeros(len(closes), dtype=int)

        dip_low = None        # lowest price seen during a dip zone
        rally_high = None     # highest price seen during a rally zone
        current_pos = 0

        for i in range(len(closes)):
            price = closes[i]

            # --- detect dip zone: drawdown exceeds threshold ---
            if dd[i] <= -self.dip_pct:
                if dip_low is None:
                    dip_low = price
                else:
                    dip_low = min(dip_low, price)

            # --- detect rally zone: drawup exceeds threshold ---
            if du[i] >= self.rally_pct:
                if rally_high is None:
                    rally_high = price
                else:
                    rally_high = max(rally_high, price)

            # --- long entry: price recovered from dip low ---
            if dip_low is not None and dip_low > 0:
                recovery = (price - dip_low) / dip_low * 100
                if recovery >= self.recovery_pct:
                    current_pos = 1
                    dip_low = None
                    rally_high = None

            # --- short entry: price pulled back from rally high ---
            if rally_high is not None and rally_high > 0:
                pullback = (rally_high - price) / rally_high * 100
                if pullback >= self.pullback_pct:
                    current_pos = -1
                    rally_high = None
                    dip_low = None

            # --- reset stale tracking ---
            # dip resolved without triggering entry
            if dip_low is not None and dd[i] > 0:
                dip_low = None
            # rally resolved without triggering entry
            if rally_high is not None and du[i] < 0:
                rally_high = None

            positions[i] = current_pos

        out["position"] = positions
        return out
