"""Drawdown Scaling strategy – scale position size with dip depth.

Instead of a fixed-size position, this strategy scales in proportionally
to how deep the drawdown is.  Bigger dip = bigger long.  Think of it as
an automated DCA / grid that gets more aggressive at deeper discounts.

Mirror logic applies for shorts on rallies from the rolling low.
"""

import numpy as np
import pandas as pd


class Strategy:
    """Drawdown-proportional position sizing.

    The position size ramps linearly from 0 to *max_scale* as the
    drawdown deepens from 0 % to *full_scale_pct* %.

    Parameters
    ----------
    lookback : int
        Rolling window for the high/low channel.
    dip_entry_pct : float
        Minimum drawdown % to start entering long (e.g. 0.5).
    rally_entry_pct : float
        Minimum rally % to start entering short.
    full_scale_pct : float
        Drawdown/rally % at which position reaches *max_scale*.
    max_scale : float
        Maximum position multiplier (e.g. 2.0 = 2x normal size).
    """

    def __init__(
        self,
        lookback: int = 60,
        dip_entry_pct: float = 0.5,
        rally_entry_pct: float = 0.5,
        full_scale_pct: float = 3.0,
        max_scale: float = 2.0,
    ):
        self.lookback = int(lookback)
        self.dip_entry_pct = float(dip_entry_pct)
        self.rally_entry_pct = float(rally_entry_pct)
        self.full_scale_pct = float(full_scale_pct)
        self.max_scale = float(max_scale)

    def __repr__(self) -> str:
        return (
            f"DrawdownScale(lookback={self.lookback}, "
            f"dip_entry={self.dip_entry_pct}%, "
            f"rally_entry={self.rally_entry_pct}%, "
            f"full_scale={self.full_scale_pct}%, "
            f"max={self.max_scale}x)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute drawdown-scaled positions.

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
            - ``position``      – scaled position (float, not just ±1)
        """
        if "close" not in df.columns:
            raise ValueError("Missing required column: 'close'")

        out = df.copy()

        out["rolling_high"] = out["close"].rolling(self.lookback, min_periods=1).max()
        out["rolling_low"] = out["close"].rolling(self.lookback, min_periods=1).min()

        # Drawdown from high (negative %), drawup from low (positive %)
        out["drawdown_pct"] = (
            (out["close"] - out["rolling_high"]) / out["rolling_high"] * 100
        )
        out["drawup_pct"] = (
            (out["close"] - out["rolling_low"]) / out["rolling_low"] * 100
        )

        dd = out["drawdown_pct"].values
        du = out["drawup_pct"].values
        positions = np.zeros(len(out))

        scale_range = self.full_scale_pct - self.dip_entry_pct

        for i in range(len(out)):
            depth = -dd[i]   # positive when below rolling high
            height = du[i]   # positive when above rolling low

            long_scale = 0.0
            short_scale = 0.0

            # Scale long position: deeper dip = bigger long
            if depth >= self.dip_entry_pct and scale_range > 0:
                t = min((depth - self.dip_entry_pct) / scale_range, 1.0)
                long_scale = t * self.max_scale

            # Scale short position: bigger rally = bigger short
            if height >= self.rally_entry_pct and scale_range > 0:
                t = min((height - self.rally_entry_pct) / scale_range, 1.0)
                short_scale = t * self.max_scale

            # Net position: whichever signal is active
            # (both rarely fire at the same time)
            if long_scale > 0 and short_scale > 0:
                # Conflicting — take the stronger signal
                positions[i] = long_scale if long_scale >= short_scale else -short_scale
            elif long_scale > 0:
                positions[i] = long_scale
            elif short_scale > 0:
                positions[i] = -short_scale
            else:
                # No signal — hold previous position
                positions[i] = positions[i - 1] if i > 0 else 0.0

        out["position"] = positions
        return out
