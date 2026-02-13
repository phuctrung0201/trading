"""Moving-average crossover (MACROSS) strategy using EMA.

Uses pandas_ta for EMA calculation, which matches TradingView / OKX charts
(SMA-seeded, α = 2 / (period + 1)).
"""

import os
import pandas as pd

# Avoid numba cache failures in some Python environments during pandas_ta import.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import pandas_ta as ta

from logger import log
from strategy.action import Position


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
        self.last_position: Position = Position.flat()

        if self.short >= self.long:
            raise ValueError(
                f"short period ({self.short}) must be less than long period ({self.long})"
            )

        # Incremental EMA state for step()
        self._alpha_s = 2.0 / (self.short + 1)
        self._alpha_l = 2.0 / (self.long + 1)
        self._ema_s: float = 0.0
        self._ema_l: float = 0.0
        self._prev_diff: float = 0.0
        self._position: int = 0
        self._bar_count: int = 0
        self._close_buffer: list[float] = []

    def __repr__(self) -> str:
        return f"MACross(short={self.short}, long={self.long}, source={self.source})"

    def step(self, close: float) -> int:
        """Incrementally update EMAs and return position (+1/-1/0).

        Matches the SMA-seeded EMA used by pandas_ta / TradingView.
        """
        self._bar_count += 1

        # Accumulate prices until long EMA can be seeded
        if self._bar_count <= self.long:
            self._close_buffer.append(close)

        # Seed / update short EMA
        if self._bar_count < self.short:
            return 0
        elif self._bar_count == self.short:
            self._ema_s = sum(self._close_buffer) / self.short
        else:
            self._ema_s = self._alpha_s * close + (1 - self._alpha_s) * self._ema_s

        # Seed / update long EMA
        if self._bar_count < self.long:
            return 0
        elif self._bar_count == self.long:
            self._ema_l = sum(self._close_buffer) / self.long
            self._close_buffer = []  # free memory
            self._prev_diff = self._ema_s - self._ema_l
            return self._position  # stays 0 — no crossover on seed bar
        else:
            self._ema_l = self._alpha_l * close + (1 - self._alpha_l) * self._ema_l

        # Crossover detection
        diff = self._ema_s - self._ema_l
        if diff > 0 and self._prev_diff <= 0:
            self._position = 1
        elif diff < 0 and self._prev_diff >= 0:
            self._position = -1
        self._prev_diff = diff

        return self._position

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

        # Build Position for the latest bar
        last_pos = int(out["position"].iloc[-1])
        if last_pos == 1:
            self.last_position = Position.long()
        elif last_pos == -1:
            self.last_position = Position.short()
        else:
            self.last_position = Position.flat()

        # Print EMA values when a crossover fires on the latest bar
        last_signal = out["signal"].iloc[-1]
        if last_signal != 0:
            log.debug(f"MACross({self.short}/{self.long}) → {self.last_position.side.name}  "
                     f"ema_short={out['ema_short'].iloc[-1]:.4f}  "
                     f"ema_long={out['ema_long'].iloc[-1]:.4f}")

        return out
