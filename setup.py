from entry.drawdown import DrawdownPositionSize
from signal.macross import MACross

pair = "ETH/USDT"
instrument = "ETH-USDT-SWAP"
step = "1m"
cap = 100
leverage = "10"

# Historical period for backtesting
start = "2025-01-10T00:00:00Z"
end = "2025-01-20T00:00:00Z"

# How far back to fetch candles when live trading starts
preload_duration = "6h"

# Scale position size down as portfolio drawdown deepens.
# At 8% drawdown, reduce to 0.4% size; at 15%, to 0.25%; at 20%, stop trading.
entry = DrawdownPositionSize(
    signal=MACross(short=15, long=30),
    thresh_hold={
        0: 0.5,
        2: 0.04,
        6: 0.025,
        10: 0
    },
    window=100,
)
