"""Stop-loss strategy that closes positions when loss exceeds a threshold."""

import pandas as pd


class Strategy:
    """Stop-loss risk management strategy.

    Monitors open positions and forces them closed when the unrealised
    loss from the entry price exceeds a configurable percentage.

    Parameters
    ----------
    percent : float | int
        Maximum allowed loss as a percentage of the entry price.
        E.g. ``20`` means close the position if it loses 20 %.
    """

    def __init__(self, percent: float | int = 10):
        self.percent = float(percent)

    def __repr__(self) -> str:
        return f"StopLoss(percent={self.percent})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stop-loss logic to existing position signals.

        If the DataFrame already contains a ``position`` column (e.g. from
        a prior strategy), this method overrides positions to 0 whenever
        the stop-loss threshold is breached.  If no ``position`` column
        exists, the DataFrame is returned unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame, optionally with a ``position`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with positions adjusted for stop-loss exits.
        """
        out = df.copy()

        if "position" not in out.columns:
            return out

        threshold = self.percent / 100.0
        positions = out["position"].values.copy()
        closes = out["close"].values

        entry_price = None
        current_pos = 0
        stopped_out = False
        stopped_out_direction = 0

        for i in range(len(positions)):
            original_pos = positions[i]
            price = closes[i]

            # After a stop-loss exit, stay flat until the base strategy
            # changes direction (e.g. a new crossover signal).
            if stopped_out:
                if original_pos != stopped_out_direction:
                    # Base strategy changed direction — allow re-entry
                    stopped_out = False
                    entry_price = price
                    current_pos = original_pos
                else:
                    # Same direction as what was stopped out — stay flat
                    positions[i] = 0
                    continue

            # Detect new position or position change
            elif original_pos != current_pos:
                if original_pos != 0:
                    entry_price = price
                else:
                    entry_price = None
                current_pos = original_pos
                continue

            # Check stop-loss for active position
            if current_pos != 0 and entry_price is not None:
                if current_pos == 1:  # long
                    loss_pct = (entry_price - price) / entry_price
                else:  # short
                    loss_pct = (price - entry_price) / entry_price

                if loss_pct >= threshold:
                    positions[i] = 0
                    stopped_out = True
                    stopped_out_direction = current_pos
                    current_pos = 0
                    entry_price = None

        out["position"] = positions
        return out
