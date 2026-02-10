"""Equity Drawdown Guard – overlay that pauses trading during losing streaks.

This is applied **after** a base strategy has produced a ``position``
column.  It simulates the equity curve from those positions and
overrides the position to 0 (flat) whenever equity draws down more
than a configurable threshold from its peak.

Trading resumes once equity recovers to within *resume_pct* of the
peak, or immediately on a new equity high.
"""

import numpy as np
import pandas as pd


class Strategy:
    """Equity-curve drawdown guard (overlay).

    Parameters
    ----------
    max_dd_pct : float
        Maximum allowed equity drawdown (%) before pausing.
        E.g. ``5.0`` means go flat when equity drops 5 % from its peak.
    resume_pct : float
        Equity must recover to within this % of the peak to resume.
        E.g. ``2.0`` means resume when drawdown shrinks back to −2 %.
        Set to ``0.0`` to require a new equity high before resuming.
    """

    def __init__(self, max_dd_pct: float = 5.0, resume_pct: float = 2.0):
        self.max_dd_pct = float(max_dd_pct)
        self.resume_pct = float(resume_pct)

    def __repr__(self) -> str:
        return (
            f"EquityGuard(max_dd={self.max_dd_pct}%, "
            f"resume={self.resume_pct}%)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override positions to flat during equity drawdowns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains ``close`` and ``position``
            columns (produced by a prior strategy).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``position`` adjusted: forced to 0 while
            the guard is active.  An ``equity_guard`` column is added
            (1 = trading allowed, 0 = paused).
        """
        if "position" not in df.columns:
            return df.copy()
        if "close" not in df.columns:
            raise ValueError("Missing required column: 'close'")

        out = df.copy()

        closes = out["close"].values
        base_positions = out["position"].values.copy()
        positions = base_positions.copy()
        guard = np.ones(len(closes), dtype=int)  # 1 = trading, 0 = paused

        # Simulate equity from the base positions
        returns = np.zeros(len(closes))
        for i in range(1, len(closes)):
            if closes[i - 1] != 0:
                returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1]

        equity = 1.0       # normalised equity starting at 1.0
        peak = 1.0
        paused = False

        for i in range(len(closes)):
            # Update equity based on the PREVIOUS bar's position
            if i > 0:
                equity *= 1.0 + base_positions[i - 1] * returns[i]

            peak = max(peak, equity)
            dd_pct = (equity - peak) / peak * 100  # negative when below peak

            if paused:
                # Check if equity recovered enough to resume
                if dd_pct >= -self.resume_pct:
                    paused = False
                else:
                    positions[i] = 0
                    guard[i] = 0
                    continue
            else:
                # Check if we need to pause
                if dd_pct <= -self.max_dd_pct:
                    paused = True
                    positions[i] = 0
                    guard[i] = 0
                    continue

        out["position"] = positions
        out["equity_guard"] = guard
        return out
