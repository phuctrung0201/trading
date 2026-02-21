# Trading

Algorithmic trading framework for backtesting and paper futures execution on OKX. Strategies consume a real-time trade stream or historical data, with metrics recorded to ClickHouse and visualized in Grafana.

## Project Structure

```
trading/
├── Makefile
├── supervisord.conf
├── docker-compose.yaml              # ClickHouse + Grafana stack
├── requirements.txt
├── config/
│   ├── app/logger.yaml               # Log level
│   ├── okx/client.yaml               # OKX API credentials
│   ├── clickhouse/client.yaml        # ClickHouse connection
│   ├── backtest/*.yaml               # Per-setup backtest configs
│   └── paper/*.yaml                  # Per-setup paper trading configs
├── src/
│   ├── app/
│   │   ├── config.py                 # YAML config loader
│   │   ├── provider.py               # AppProvider — wires all dependencies
│   │   └── logger.py                 # Structured logging
│   ├── strategy/
│   │   └── adapter.py                # BaseStrategy — signal, execution, measurement, reconciliation
│   ├── crossma/
│   │   ├── strategy.py               # CrossMAStrategy — EMA crossover
│   │   └── registry.py               # Strategy registry
│   ├── drawdown/
│   │   └── strategy.py               # DrawdownStrategy — drawdown-aware position sizing
│   ├── ema/
│   │   └── indicator.py              # EMA calculator
│   ├── exchange/
│   │   ├── adapter.py                # ExchangeAdapter interface
│   │   ├── dto.py                    # MarketTrade, Position, OpenResult
│   │   ├── position.py               # PositionTracker — in-memory equity & PnL
│   │   └── simulator.py              # SimulateExchange for backtests
│   ├── okx/
│   │   ├── auth.py                   # OKX API auth & signing
│   │   ├── client.py                 # OkxClient — assembles auth, trading, account
│   │   ├── trading.py                # Order placement, close position
│   │   ├── account.py                # Balance, positions, leverage
│   │   └── exchange.py               # OkxExchange — live adapter with rate limiting
│   ├── clickhouse/
│   │   ├── client.py                 # ClickHouseClient — buffered Parquet inserts
│   │   ├── schema.py                 # DDL and PyArrow schemas for trade_event & ops
│   │   ├── measurement.py            # TradeMeasurement, TradeEventMeasurement, OpsMeasurement
│   │   └── recorder.py               # ClickHouseRecorder — routes measurements to tables
│   ├── backtest/
│   │   ├── main.py                   # Backtest entry point
│   │   ├── app.py                    # BacktestApp
│   │   └── data.py                   # CSV download/load for trade history
│   └── paper/
│       ├── main.py                   # Paper trading entry point
│       └── app.py                    # PaperApp — live paper trading loop
└── grafana/
    ├── dashboards/trade.json         # Pre-built Grafana dashboard
    └── provisioning/
        ├── datasources/clickhouse.yaml
        └── dashboards/dashboards.yaml
```

## Architecture

The framework is built around **adapters**, **strategies**, and **recorders**, wired together by `AppProvider`.

### Exchange Adapters

`ExchangeAdapter` decouples strategies from exchange mechanics. Two implementations:

- **`OkxExchange`** — live execution on OKX with leverage management, contract-size conversion, margin-error retry, rate limiting, and parallel history fetch.
- **`SimulateExchange`** — delegates to `PositionTracker` for in-memory backtesting.

Both use `PositionTracker` for equity and unrealized PnL tracking.

### Strategies

Strategies consume `MarketTrade` objects through `ack(trade)`. `BaseStrategy` in `src/strategy/adapter.py` provides:

- **`TradeAggregator`** — buckets raw trades into time windows and emits a close price per bucket.
- **EMA signal** — computes short/long EMA and delegates to `compute_signal()`.
- **Execution** — opens/closes positions on signal flips.
- **Measurement** — emits `TradeMeasurement` per bucket and `TradeEventMeasurement` on open/close/resize/error.
- **Reconciliation** — periodically compares local state with exchange and records `OpsMeasurement`.

Concrete strategies:

- **`CrossMAStrategy`** — EMA crossover; goes long when short EMA > long EMA, short otherwise.
- **`DrawdownStrategy`** — extends crossover with rolling drawdown window and tiered position scaling.

### Recording

`ClickHouseRecorder` routes measurements to ClickHouse tables:

- **`trade_event`** — periodic metrics (event=NULL) and discrete trade events (open/close/resize/error) with fill price, PnL, signal, and reason.
- **`ops`** — operational telemetry: API latency, candle lag, write buffer size, reconciliation diffs, errors.

`ClickHouseClient` buffers rows in-memory and flushes as Parquet to ClickHouse's HTTP interface.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Configure credentials in `config/okx/client.yaml`:

```yaml
api_key: "your-okx-api-key"
secret_key: "your-okx-secret-key"
passphrase: "your-okx-passphrase"
demo: true
```

3. Configure ClickHouse connection in `config/clickhouse/client.yaml`:

```yaml
enabled: true
url: http://localhost:8123
database: trading
user: default
password: "trading"
```

## Configuration

Each setup lives in its own YAML file under `config/<mode>/<name>.yaml`.

Example `config/paper/sol-drawdown.yaml`:

```yaml
exchange:
  instrument: SOL-USDT-SWAP
  steps: 5m
  leverage: 1

depth:
  hour: 0
  minute: 30

crossma:
  short_length: 400
  long_length: 500

drawdown:
  window: 1440
  threshold_scale_map:
    0.0: 1.0
    0.02: 0.2
    0.05: 0.01
```

| Section | Key | Description |
|---------|-----|-------------|
| `exchange` | `instrument` | OKX instrument ID |
| | `steps` | Candle/bucket interval (e.g. `5m`, `1h`) |
| | `leverage` | Futures leverage |
| `depth` | `hour`, `minute` | History depth for warm-up preload |
| `crossma` | `short_length`, `long_length` | EMA periods |
| `drawdown` | `window` | Rolling equity peak lookback (candles) |
| | `threshold_scale_map` | Drawdown % → position scale factor |

## Usage

### Backtest

```bash
make backtest            # run all configs in config/backtest/
make backtest-one SETUP=sol-drawdown   # run a single setup
```

Downloads trade history from OKX, runs the drawdown strategy with simulated execution, and records metrics to ClickHouse.

### Paper Trading

```bash
make paper               # start as supervised background process
make paper-status        # check process status
make paper-logs          # tail stdout/stderr logs
make paper-stop          # stop process and shut down supervisord
```

Connects to OKX (demo mode by default), preloads recent trade history for strategy warm-up, then streams real-time trades and executes via the drawdown strategy.

### Observability (ClickHouse + Grafana)

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

- **ClickHouse** on `localhost:8123` — time-series store for trade measurements and operational telemetry.
- **Grafana** on `localhost:3001` — pre-provisioned with a ClickHouse datasource and a trade dashboard.

Each session writes data tagged with a unique `session_id` and `setup` name, so you can compare runs side by side in Grafana.

## Dependencies

- requests
- pyyaml
- supervisor
- pyarrow
