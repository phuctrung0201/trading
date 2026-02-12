from strategy.drawdown import DrawdownPositionSize
from signal.macross import MACross

# OKX API credentials
okx_api_key = "your-api-key"
okx_secret_key = "your-secret-key"
okx_passphrase = "your-passphrase"
okx_demo = True

instrument = "ETH-USDT-SWAP"
step = "1m"
cap = 100
leverage = 10

# Historical period for backtesting
start = "2025-01-01T00:00:00Z"
end = "2025-04-01T00:00:00Z"

# How far back to fetch candles when live trading starts
preload_duration = "6h"

# Scale position size down as portfolio drawdown deepens.
# At 2% drawdown, reduce to 4% size; at 6%, to 2.5%; at 10%, re-evaluate signals.
strategy = DrawdownPositionSize(
    # Strategy will evaluate signals to find the best sharp ratio signals.
    signals=[
        MACross(short=15, long=17),
        MACross(short=10, long=17),
        MACross(short=5, long=17),
    ],
    size={
        0: 0.5,
        0.04: 0.04,
        0.06: 0.02,
    },
    reevaluate_threshold=0.1,
    drawdown_window=500,
    sharpe_window=1440,
)
