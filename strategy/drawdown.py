"""Drawdown-aware strategy that selects the best signal and scales position size."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from strategy.position import Position


class DrawdownPositionSize:
    """Evaluate multiple signals, pick the best Sharpe, and scale by drawdown.

    The strategy evaluates all provided signals and selects the one with
    the best rolling Sharpe ratio.  When the portfolio drawdown deepens
    beyond *reevaluate_threshold*, the signal choice is re-evaluated to
    adapt to changing market conditions.

    Position size is then scaled according to *size* thresholds based
    on the current drawdown level.

    Parameters
    ----------
    signals : list
        Strategy instances that implement ``generate_signals(df)``.
        The strategy will evaluate all signals and select the one
        with the best Sharpe ratio.
    size : dict[float, float]
        Mapping of *drawdown-fraction → position-scale*.  Keys are
        drawdown levels as decimals (e.g. ``0.04`` means 4 %) and
        values are the multiplier applied to the position when the
        drawdown reaches that level.  Higher drawdown thresholds take
        priority.

        Example::

            {
                0: 0.5,       # no drawdown      → 50 % size
                0.04: 0.04,   # at 4 % drawdown  → 4 % size
                0.06: 0.02,   # at 6 % drawdown  → 2 % size
            }
    reevaluate_threshold : float
        Drawdown fraction at which the signal selection is
        re-evaluated.  Below this level the initially chosen signal
        is held; at or above it the best current signal is picked.
    window : int
        Rolling lookback window for the equity peak.  Only the last
        *window* equity values are considered when determining the
        peak, so old highs expire over time.
    sharpe_window : int
        Rolling lookback window used to compute each signal's Sharpe
        ratio for selection purposes.
    """

    def __init__(
        self,
        signals: list,
        size: dict[int | float, float],
        drawdown_window: int,
        reevaluate_threshold: float = 0.1,
        sharpe_window: int = 1440,
    ) -> None:
        self.signals = signals
        self.drawdown_window = drawdown_window
        self.reevaluate_threshold = reevaluate_threshold
        self.sharpe_window = sharpe_window
        self.last_position: Position = Position.flat()
        # Sort descending so the highest (most severe) threshold matches first.
        self.thresh_hold: dict[float, float] = {
            float(k): v for k, v in sorted(size.items(), reverse=True)
        }

    def __repr__(self) -> str:
        return (
            f"DrawdownPositionSize(signals={len(self.signals)}, "
            f"thresh_hold={self.thresh_hold}, "
            f"reevaluate_threshold={self.reevaluate_threshold})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate signals, pick best Sharpe, and apply drawdown scaling.

        At each bar the position from the currently selected signal is
        used.  Signal selection is re-evaluated whenever the running
        drawdown exceeds *reevaluate_threshold* or on the first bar.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least a ``close`` column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the ``position`` column set to the
            scaled position of the best signal.
        """
        # -- Pre-compute signal outputs and rolling Sharpe ratios -----------
        signal_results = [s.generate_signals(df) for s in self.signals]

        rolling_sharpes = []
        for sig_df in signal_results:
            if "position" not in sig_df.columns:
                rolling_sharpes.append(np.zeros(len(df)))
                continue
            close_rets = sig_df["close"].pct_change().fillna(0)
            strat_rets = sig_df["position"].shift(1).fillna(0) * close_rets
            rmean = strat_rets.rolling(
                window=self.sharpe_window, min_periods=1
            ).mean()
            rstd = strat_rets.rolling(
                window=self.sharpe_window, min_periods=1
            ).std()
            sharpe = (rmean / rstd).fillna(0).replace([np.inf, -np.inf], 0)
            rolling_sharpes.append(sharpe.values)

        sharpe_matrix = np.column_stack(rolling_sharpes)
        all_positions = np.column_stack(
            [sig_df["position"].values.astype(float) for sig_df in signal_results]
        )

        # -- Bar-by-bar simulation -----------------------------------------
        out = df.copy()
        n = len(df)
        closes = out["close"].values
        positions = np.zeros(n)

        equity = 1.0  # normalised starting equity
        drawdown_pct = 0.0
        equity_history: deque[float] = deque([equity], maxlen=self.drawdown_window)
        peak = equity
        current_signal_idx = 0

        for i in range(n):
            if i > 0:
                # Bar return based on the position held *before* this bar
                ret = (
                    (closes[i] - closes[i - 1]) / closes[i - 1]
                    if closes[i - 1] != 0
                    else 0.0
                )
                equity *= 1.0 + positions[i - 1] * ret
                equity_history.append(equity)
                peak = max(equity_history)
                drawdown_pct = (peak - equity) / peak

            # TODO: handle reevaluate — consider suppressing trades or
            #       switching signal when no signal has a good Sharpe ratio.
            # Re-evaluate signal on the first bar or when drawdown is deep
            if i == 0 or drawdown_pct >= self.reevaluate_threshold:
                current_signal_idx = int(np.argmax(sharpe_matrix[i]))

            positions[i] = all_positions[i, current_signal_idx]

            # Find the first (highest) threshold that has been breached
            scale = 1.0
            for dd_level, dd_scale in self.thresh_hold.items():
                if drawdown_pct >= dd_level:
                    scale = dd_scale
                    break

            positions[i] *= scale

        out["position"] = positions

        # Build Entry for the latest bar
        self.last_position = Position.from_raw(float(positions[-1]))

        return out
