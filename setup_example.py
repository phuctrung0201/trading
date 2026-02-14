from strategy.drawdown import DrawdownPositionSize
from signal.macross import MACross
from client import okx
import logger

# Log level: DEBUG, INFO, WARN, ERROR, SILENT
log_level = "INFO"
logger.configure(log_level)

# OKX API credentials
okx_api_key = "your-api-key"
okx_secret_key = "your-secret-key"
okx_passphrase = "your-passphrase"
okx_demo = True

instrument = "ETH-USDT-SWAP"
step = "1m"
default_cap = 100.0
leverage = 10

# Historical period for backtesting
start = "2025-01-01T00:00:00Z"
end = "2025-04-01T00:00:00Z"

# How far back to fetch candles when live trading starts
preload_duration = "6h"

# Monitor / InfluxDB (optional)
influx_enabled = False
influx_url = "http://localhost:8086"
influx_org = "trading"
influx_bucket = "trading"
influx_token = "your-influx-token"


def okx_client() -> okx.Client:
    return okx.Client(
        api_key=okx_api_key,
        secret_key=okx_secret_key,
        passphrase=okx_passphrase,
        demo=okx_demo,
    )


cap = default_cap

# Scale position size down as portfolio drawdown deepens.
# At 2% drawdown, reduce to 4% size; at 6%, to 2.5%; at 10%, re-evaluate signals.
strategy = DrawdownPositionSize(
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
    drawdown_window=500,
    equity=cap,
)
