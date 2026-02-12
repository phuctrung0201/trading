"""Drawdown-aware entry overlay that scales position size during drawdowns."""

from __future__ import annotations

from collections import deque

import pandas as pd


class DrawdownPositionSize:
    """Scale or suppress entries based on running portfolio drawdown.

    Before a new position is opened, the current drawdown from the equity
    peak is checked against a set of thresholds.  If the drawdown exceeds
    a threshold, the position size is scaled by the corresponding factor,
    effectively adding a tighter stoploss.  If the factor is ``0`` the
    position is suppressed entirely.

    Parameters
    ----------
    signal
        Strategy instance that implements ``generate_signals(df)``.
    thresh_hold : dict[int | float, float]
        Mapping of *drawdown-percentage → position-scale*.  Keys are
        drawdown levels (e.g. ``8`` means 8 %) and values are the
        multiplier applied to the position when the drawdown reaches
        that level.  Higher drawdown thresholds take priority.

        Example::

            {
                8:  0.004,   # at 8 % drawdown  → scale to 0.4 %
                15: 0.0025,  # at 15 % drawdown → scale to 0.25 %
                20: 0,       # at 20 % drawdown → stop trading
            }
    window : int
        Rolling lookback window for the equity peak.  Only the last
        *window* equity values are considered when determining the
        peak, so old highs expire over time.
    """

    def __init__(
        self,
        signal,
        thresh_hold: dict[int | float, float],
        window: int,
    ) -> None:
        self.signal = signal
        self.window = window
        # Sort descending so the highest (most severe) threshold matches first.
        self.thresh_hold: dict[float, float] = {
            float(k): v for k, v in sorted(thresh_hold.items(), reverse=True)
        }

    def __repr__(self) -> str:
        return f"DrawdownStoploss(thresh_hold={self.thresh_hold})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply drawdown-based scaling to an existing ``position`` column.

        The method tracks a simulated equity curve bar-by-bar using the
        ``close`` prices and the incoming ``position`` column.  Whenever
        the running drawdown from peak equity exceeds a threshold, the
        position for that bar is multiplied by the corresponding scale
        factor.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame that already contains a ``position`` column
            (e.g. produced by a prior strategy).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the ``position`` column adjusted, and an
            extra ``drawdown_pct`` column showing the running drawdown.
        """
        out = df.copy()

        if "position" not in out.columns:
            return out

        closes = out["close"].values
        positions = out["position"].values.copy().astype(float)

        equity = 1.0  # normalised starting equity
        drawdown_pct = 0.0

        # Track equity history for rolling peak.
        equity_history: deque[float] = deque([equity], maxlen=self.window)
        peak = equity

        for i in range(len(positions)):
            if i > 0:
                # Bar return based on the position held *before* this bar
                ret = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] != 0 else 0.0
                equity *= 1.0 + positions[i - 1] * ret
                equity_history.append(equity)
                peak = max(equity_history)
                drawdown_pct = ((peak - equity) / peak) * 100.0

            # Find the first (highest) threshold that has been breached
            scale = 1.0
            for dd_level, dd_scale in self.thresh_hold.items():
                if drawdown_pct >= dd_level:
                    scale = dd_scale
                    break

            positions[i] *= scale

        out["position"] = positions
        return out
