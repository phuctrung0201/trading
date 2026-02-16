# Trading

Algorithmic trading framework for backtesting and live futures execution on OKX. Strategies run against a real-time candle stream or historical data, with metrics recorded to InfluxDB and visualized in Grafana.

## Project Structure

```
trading/
├── trade.py                        # Live trading entry point
├── backtest.py                     # Backtest entry point
├── setup.yaml                      # Runtime config (credentials, strategy, timeframe)
├── setup.example.yaml              # Example config with placeholder values
├── docker-compose.yaml             # InfluxDB + Grafana stack
├── requirements.txt
├── src/
│   ├── app/
│   │   ├── core.py                 # CoreApp — shared init for config, logger, clients
│   │   ├── trade.py                # TradeApp — live trading application
│   │   ├── backtest.py             # BacktestApp — backtesting application
│   │   ├── adapter.py              # Exchange & measurement adapter interfaces
│   │   ├── config.py               # YAML config loader and typed dataclasses
│   │   └── logger.py               # Structured logging
│   ├── strategy/
│   │   ├── noaction.py             # NoActionStrategy — base pass-through strategy
│   │   ├── crossma.py              # CrossMAStrategy — EMA crossover with mark-to-market
│   │   └── drawdown.py             # DrawdownStrategy — drawdown-aware position sizing
│   ├── indicator/
│   │   └── ema.py                  # EMA calculator
│   ├── measurement/
│   │   └── trade.py                # TradeMeasurement dataclass (equity, drawdown, Sharpe)
│   └── client/
│       ├── okx.py                  # OKX REST client (market data, trading, account)
│       ├── influxdb.py             # InfluxDB v2 writer with buffered background flush
│       └── ohclv.py                # OHCLV candle dataclass
└── grafana/
    ├── dashboards/                 # Pre-built Grafana dashboard JSON
    └── provisioning/               # Grafana datasource and dashboard provisioning
```

## Architecture

The framework is built around three abstractions — **adapters**, **strategies**, and **clients** — wired together by an application layer.

### Adapters

Adapters decouple strategies from external systems. Two interfaces are defined in `src/app/adapter.py`:

- **ExchangeAdapter** — abstract interface for opening/closing positions and querying balances.
  - `OkxExchangeAdapter` — live execution on OKX with leverage management, contract-size conversion, margin-error retry, and exponential backoff.
  - `SimulateAdapter` — in-memory position tracker used during backtests.
- **MeasurementAdapter** — abstract interface for recording strategy metrics.
  - `InfluxAdapter` — writes `TradeMeasurement` data points to InfluxDB, tagged by session ID.

### Strategies

Strategies consume OHCLV candles one at a time through the `ack(candle)` method.

- **NoActionStrategy** — base class; logs each candle and does nothing.
- **CrossMAStrategy** — EMA crossover strategy. Computes short and long EMAs over a sliding window of close prices. When the short EMA crosses above the long, it opens a long; when it crosses below, it opens a short. Tracks equity via bar-by-bar mark-to-market, computes rolling drawdown and annualized Sharpe ratio, and emits a `TradeMeasurement` on every candle.
- **DrawdownStrategy** — extends `CrossMAStrategy` with drawdown-aware position sizing. Maintains a rolling equity window and scales position size down through configurable threshold/scale tiers as drawdown deepens.

### Indicator

- **EMA** — computes an exponential moving average over an arbitrary-length value array.

### Clients

- **OkxClient** — signed REST client for OKX. Handles order placement, position close, leverage, mark price, instrument info, and candle history. Provides two generators: `stream_history` for backtesting (paginated history-candles endpoint) and `stream_prices` for live trading (polling confirmed candles).
- **InfluxDBClient** — writes line-protocol batches to InfluxDB v2 over HTTP. Uses a background `InfluxWorker` thread that buffers points and flushes periodically or when the buffer fills.

### Application Layer

- **CoreApp** — initializes config, logger, OKX client, and InfluxDB client.
- **TradeApp** — extends `CoreApp` for live trading. Preloads recent candles for strategy warm-up, then streams real-time candles from OKX and feeds them into the selected strategy.
- **BacktestApp** — extends `CoreApp` for backtesting. Uses `SimulateAdapter` for position tracking and streams historical candles from OKX through the strategy.

## Setup

1. Copy the example config:

```bash
cp setup.example.yaml setup.yaml
```

2. Fill in your OKX API credentials and (optionally) InfluxDB token in `setup.yaml`:

```yaml
okx:
  api_key: "your-okx-api-key"
  secret_key: "your-okx-secret-key"
  passphrase: "your-okx-passphrase"
```

> `setup.yaml` contains secrets and should not be committed. Only `setup.example.yaml` is safe to share.

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All parameters live in `setup.yaml`:

```yaml
log_level: INFO

influx:
  enabled: true
  url: http://localhost:8086
  org: trading
  bucket: trading
  token: <your-influxdb-token>

okx:
  api_key: <your-okx-api-key>
  secret_key: <your-okx-secret-key>
  passphrase: <your-okx-passphrase>

crossma:
  short_length: 15          # Short EMA period
  long_length: 200          # Long EMA period
  equity: auto              # "auto" fetches balance from OKX; or set a fixed number

drawdown:
  window: 1440              # Rolling equity peak lookback (candles)
  threshold_scale_map:      # Drawdown % → position scale factor
    "0": 1.0                # No drawdown → 100% size
    "0.1": 0.01             # 10% drawdown → 1% size
    "0.2": 0.001            # 20% drawdown → 0.1% size

trade:
  demo: true                # true = OKX demo trading; false = real funds
  instrument: ETH-USDT-SWAP # OKX instrument ID
  steps: 5m                 # Candle interval
  preload: 1d               # History to preload for warm-up
  leverage: 1               # Futures leverage
  strategy: drawdown        # "crossma" or "drawdown"

backtest:
  start: "2026-02-13T00:00:00Z"
  end: "2026-02-14T00:00:00Z"
```

## Usage

### Backtest

```bash
make backtest
```

Streams historical candles from OKX for the configured time window, runs the drawdown strategy with simulated execution, and records metrics to InfluxDB (if enabled).

### Live Trading

```bash
make trade
```

Starts `trade.py` as a supervised background process via `supervisord`. The process auto-restarts on crash and logs to `.supervisor/trade.stdout.log` and `.supervisor/trade.stderr.log`.

Connects to OKX (demo mode by default), preloads recent candle history for strategy warm-up, then polls for new confirmed candles and executes trades based on strategy signals.

```bash
make trade-status   # check if the process is running
make trade-logs     # tail stdout and stderr logs
make trade-stop     # stop the process and shut down supervisord
```

### Observability (InfluxDB + Grafana)

Start the metrics stack:

```bash
make monitor
```

Stop or tail logs:

```bash
make monitor-down
make monitor-logs
```

This brings up:

- **InfluxDB** on `localhost:8086` — time-series store for trade measurements (equity, drawdown, Sharpe ratio, position size, EMA values).
- **Grafana** on `localhost:3001` — pre-provisioned with an InfluxDB datasource and a backtest dashboard.

Each backtest or live session writes data points tagged with a unique `session_id`, so you can compare runs side by side in Grafana.

## Dependencies

- requests
- pyyaml
- supervisor
