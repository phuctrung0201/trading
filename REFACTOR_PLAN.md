# Refactor Plan (Expectation Alignment)

## Target Structure
The project keeps runtime entrypoints at the repository root and places implementation code under `src/`.
Inside `src/`, only these L2 folders are used: `app`, `client`, `indicator`, `measurement`, `strategy`.

```text
trading/
  setup.yaml   # app runtime configuration source (keep above entrypoints)
  backtest.py   # entrypoint (outside src, delegates to src/app/backtest.py)
  trade.py      # entrypoint (outside src)
  src/
    app/        # App flows and runtime wiring (backtest/trade/core)
    client/     # External integration clients (exchange/storage/transport)
    indicator/  # Pure indicator/signal calculations (no network side effects)
    measurement/ # Measurement models and metrics payload definitions
    strategy/   # Trading decision logic (action/risk/position sizing)
```

## Target File
**`src/app`**
- `config.py`
  - app config loading module
  - load and validate `setup.yaml`
  - define config class
- `logger.py`
  - app logger setup module
  - define log methods
- `core.py`
  - init config
  - init logger
  - init InfluxDB client
  - init other clients
  - init strategies
- `backtest.py`
  - backtest app flow module
  - extend `core.py`
  - load historical candles via OKX client in app layer
  - feed historical candles to `CrossMAStrategy`
  - expose callable app function for root entrypoint delegation
- `trade.py`
  - live-trade app flow module
  - extend `core.py`
  - own OKX exchange adapter for live execution

**`src/client`**
- `ohclv.py`
  - define OHCLV candle format
  - load and normalize OHCLV records
- `okx.py`
  - OKX API wrapper
  - return prices data in OHCLV format
- `influxdb.py`
  - InfluxDB metrics sink adapter
  - manage write lifecycle handling
  - support buffered flush and async flush worker
  - define worker class for async flush processing
  - `write(...)` pushes message to buffer queue for flush

**`src/indicator`**
- `ema.py`
  - EMA indicator calculation
  - consume OHCLV price series

**`src/measurement`**
- `trade.py`
  - trade measurement payload structure
- `backtest.py`
  - backtest measurement payload structure

**`src/strategy`**
- `adapter.py`
  - define exchange adapter interface for strategy layer
  - define position model
  - define open result model
  - define open position method
  - define asset balance getter
- `noaction.py`
  - define `__init__()`
  - implement `ack(candle)` method
  - receive a candle by time
  - do nothing (no-op behavior)
- `crossma.py`
  - extend `noaction.py`
  - set `short` and `long` to fixed values in `__init__(exchange_adapter)`
  - call EMA calculation for short period and long period
  - moving-average crossover strategy logic
  - consume EMA indicator values
  - `ack(candle: OHCLV)` executes via exchange adapter and does not return action
- `drawdown.py`
  - extend `crossma.py`
  - drawdown-aware position sizing in `__init__(exchange_adapter)`
  - `ack(candle: OHCLV)` executes via exchange adapter and does not return action

## TODO
- [x] create `src/client/ohclv.py` to define OHCLV candle format
- [x] define class `OHCLV` in `src/client/ohclv.py`
- [x] define function `__init__(timestamp, open, high, low, close, volume)` in class `OHCLV`
- [x] create `src/client/okx.py` as OKX API wrapper returning prices in OHCLV format
- [x] define class `OkxClient` in `src/client/okx.py`
- [x] define function `__init__(api_key, secret_key, passphrase, demo)` in class `OkxClient`
- [x] define function `get_prices(instrument, start, end, step)` in class `OkxClient`
- [x] define function `stream_prices(instrument, step)` in class `OkxClient`
- [x] create `src/client/influxdb.py` for metrics write handling
- [x] define class `InfluxDBClient` in `src/client/influxdb.py`
- [x] define class `InfluxWorker` in `src/client/influxdb.py`
- [x] define function `__init__(url, token, org, bucket)` in class `InfluxDBClient`
- [x] define function `__init__(client, buffer_size, flush_delay)` in class `InfluxWorker`
- [x] define function `queue(measurement, fields, timestamp)` in class `InfluxWorker`
- [x] define function `write(measurement, fields, timestamp)` in class `InfluxDBClient`
- [x] make `write(measurement, fields, timestamp)` push message to buffer queue for flush
- [x] define function `flush()` in class `InfluxWorker`
- [x] define function `close()` in class `InfluxDBClient`
- [x] create `src/app/config.py`
- [x] define class `AppConfig` in `src/app/config.py`
- [x] define function `load_yaml(config_path)` in class `AppConfig`
- [x] create `src/app/logger.py`
- [x] define class `AppLogger` in `src/app/logger.py`
- [x] define function `init_logger(log_level)` in `src/app/logger.py`
- [x] define function `debug(message)` in class `AppLogger`
- [x] define function `info(message)` in class `AppLogger`
- [x] define function `warn(message)` in class `AppLogger`
- [x] define function `error(message)` in class `AppLogger`
- [x] create `src/indicator/ema.py` for EMA calculation
- [x] define class `EMA` in `src/indicator/ema.py`
- [x] define function `calculate(values)` in class `EMA`
- [x] create `src/strategy/noaction.py` with no-op `ack(candle)`
- [x] define class `NoActionStrategy` in `src/strategy/noaction.py`
- [x] define function `__init__()` in class `NoActionStrategy`
- [x] define function `ack(candle: OHCLV)` in class `NoActionStrategy`
- [x] create `src/strategy/adapter.py`
- [x] define interface `ExchangeAdapter` in `src/strategy/adapter.py`
- [x] define class `Position` in `src/strategy/adapter.py`
- [x] define class `OpenResult` in `src/strategy/adapter.py`
- [x] define function `open(position)` in class `ExchangeAdapter`
- [x] define function `get_asset(asset)` in class `ExchangeAdapter`
- [x] create `src/strategy/crossma.py`
- [x] define class `CrossMAStrategy` in `src/strategy/crossma.py`
- [x] make class `CrossMAStrategy` extend `NoActionStrategy`
- [x] define function `init_equity(exchange_adapter)` in class `CrossMAStrategy`
- [x] define function `init_short_length(length)` in class `CrossMAStrategy`
- [x] define function `init_long_length(length)` in class `CrossMAStrategy`
- [x] define function `__init__(exchange_adapter)` in class `CrossMAStrategy`
- [x] define function `is_long(short_ema, long_ema)` in class `CrossMAStrategy`
- [x] define function `is_short(short_ema, long_ema)` in class `CrossMAStrategy`
- [x] define function `ack(candle: OHCLV)` in class `CrossMAStrategy`
- [x] create `src/strategy/drawdown.py`
- [x] define class `DrawdownStrategy` in `src/strategy/drawdown.py`
- [x] make class `DrawdownStrategy` extend `CrossMAStrategy`
- [x] define function `init_drawdown(window)` in class `DrawdownStrategy`
- [x] define function `init_threshold(threshold: dict[float, float])` in class `DrawdownStrategy`
- [x] define function `__init__(exchange_adapter)` in class `DrawdownStrategy`
- [x] define function `calculate_drawdown(candle)` in class `DrawdownStrategy`
- [x] define function `scale_position(drawdown)` in class `DrawdownStrategy`
- [x] define function `ack(candle: OHCLV)` in class `DrawdownStrategy`
- [x] create `src/app/core.py`
- [x] define class `CoreApp` in `src/app/core.py`
- [x] define function `init_config(config_path)` in class `CoreApp`
- [x] define function `init_logger(log_level)` in class `CoreApp`
- [x] define function `init_influxdb_client(config)` in class `CoreApp`
- [x] define function `init_okx_client(config)` in class `CoreApp`
- [x] define function `__init__()` in class `CoreApp`
- [x] create `src/app/backtest.py`
- [x] define class `BacktestApp` extends `CoreApp` in `src/app/backtest.py`
- [x] define class `SimulateAdapter` in `src/app/backtest.py`
- [x] implement `ExchangeAdapter` interface for class `SimulateAdapter`
- [x] define class `MeasurementAdapter` in `src/app/backtest.py`
- [x] define function `new_session_id()` in class `MeasurementAdapter`
- [x] define function `init_tag_set()` in class `MeasurementAdapter`
- [x] define function `__init__(influxdb_client)` in class `MeasurementAdapter`
- [x] define function `record(fields: dict)` in class `MeasurementAdapter`
- [x] define function `init_crossma_strategy(exchange_adapter)` in class `BacktestApp`
- [x] define function `init_drawdown_strategy(exchange_adapter)` in class `BacktestApp`
- [x] define function `__init__()` in class `BacktestApp`
- [x] update function `__init__(exchange_adapter, measurement_adapter)` in class `CrossMAStrategy`
- [x] update function `__init__(exchange_adapter, measurement_adapter)` in class `DrawdownStrategy`
- [x] update function `init_crossma_strategy(exchange_adapter, measurement_adapter)` in class `BacktestApp`
- [x] update function `init_drawdown_strategy(exchange_adapter, measurement_adapter)` in class `BacktestApp`
- [x] update function `ack(candle: OHCLV)` in class `CrossMAStrategy` to call `measurement_adapter.record(fields)`
- [x] update function `ack(candle: OHCLV)` in class `DrawdownStrategy` to call `measurement_adapter.record(fields)`
- [x] define interface `Measurable` in `src/strategy/adapter.py`
- [x] define function `to_dict()` in interface `Measurable`
- [x] update function `record(measurable: Measurable)` in class `MeasurementAdapter`
- [x] update function `ack(candle: OHCLV)` in class `CrossMAStrategy` to call `measurement_adapter.record(trade_measurement)`
- [x] update function `ack(candle: OHCLV)` in class `DrawdownStrategy` to call `measurement_adapter.record(trade_measurement)`
- [x] create `src/measurement/trade.py`
- [x] define class `TradeMeasurement` in `src/measurement/trade.py`
- [x] make class `TradeMeasurement` extend `Measurable`
- [x] define fields `equity`, `position_size`, `position_side`, `drawdown`, `sharpe_ratio` in class `TradeMeasurement`
- [x] define function `to_dict()` in class `TradeMeasurement`
- [x] define root `setup.yaml` as default app config file
- [x] define InfluxDB setup section in `setup.yaml` (`url`, `token`, `org`, `bucket`, write behavior)
- [x] define OKX setup section in `setup.yaml` (`api_key`, `secret_key`, `passphrase`, `demo`, trading instrument/timeframe)
- [x] define CrossMA setup section in `setup.yaml` (`short_length`, `long_length`, sizing/equity controls)
- [x] define Drawdown setup section in `setup.yaml` (`window`, threshold-to-scale map, risk limits)
- [x] define Trade setup section in `setup.yaml` (`mode`, `instrument`, `steps`)
- [x] define Backtest setup section in `setup.yaml` (`start`, `end`)
- [x] update `AppConfig` in `src/app/config.py` to load `setup.yaml` by default
- [x] map `setup.yaml` InfluxDB section into `AppConfig` fields used by `init_influxdb_client(config)`
- [x] map `setup.yaml` OKX section into `AppConfig` fields used by `init_okx_client(config)`
- [x] map `setup.yaml` CrossMA section into `AppConfig` fields used by `init_crossma_strategy(...)`
- [x] map `setup.yaml` Drawdown section into `AppConfig` fields used by `init_drawdown_strategy(...)`
- [x] map `setup.yaml` Trade section (`mode`, `instrument`, `steps`) into `AppConfig` fields used by `src/app/trade.py` app wiring
- [x] map `setup.yaml` Backtest section (`start`, `end`) into `AppConfig` fields used by `src/app/backtest.py` app wiring
- [x] update `CoreApp.init_config(config_path)` to use `setup.yaml` when no config path is provided
- [x] verify `InfluxDBClient` initialization uses the configured `setup.yaml` values
- [x] verify `OkxClient` initialization uses the configured `setup.yaml` values
- [x] verify `CrossMAStrategy` initialization uses the configured `setup.yaml` values
- [x] verify `DrawdownStrategy` initialization uses the configured `setup.yaml` values
- [x] verify backtest app initialization uses configured Backtest `start` and `end`
- [x] define and keep a stable callable entry in `src/app/backtest.py` (`main()`), which owns the backtest execution flow
- [x] refactor root `backtest.py` to only call `src/app/backtest.py::main()` (no backtest business logic in root entry)
- [x] initialize `BacktestApp` inside app `main()` and read runtime config from `setup.yaml`
- [x] load historical candles via app OKX client using configured `backtest.start`, `backtest.end`, trade `instrument`, and trade `steps`
- [x] feed each historical candle into `CrossMAStrategy.ack(...)` from app flow
- [x] gate Influx client setup/write path by `setup.yaml` `influx.enabled` to preserve legacy enable/disable behavior
- [x] keep output behavior parity in app flow (`summary` and `backtest.png`)
- [x] verify `python backtest.py` executes through delegated app `main()` path
- [x] verify parity for inputs (`start`, `end`, `instrument`, `steps`) between setup config and executed backtest run
- [x] verify parity for outputs (strategy execution path, measurement writes when enabled, summary, and `backtest.png`)
- [ ] create `src/app/trade.py`
- [ ] verify live trade app initialization uses configured Trade `mode`, `instrument`, and `steps`
- [ ] make `src/app/trade.py` extend `core.py`
- [ ] implement OKX exchange adapter ownership in `src/app/trade.py`
- [ ] define function `build_exchange_adapter()` in `src/app/trade.py`
- [ ] define function `build_app(core, exchange_adapter)` in `src/app/trade.py`
- [ ] define and keep a stable callable entry in `src/app/trade.py` (`main()`), which owns the live trade execution flow
- [ ] refactor root `trade.py` to only call `src/app/trade.py::main()` (no trade business logic in root entry)
- [ ] initialize `TradeApp` inside app `main()` and read runtime config from `setup.yaml`
- [ ] stream live candles via app OKX client using configured trade `instrument` and `steps`
- [ ] feed each live candle into strategy `ack(...)` from app flow
- [ ] verify `python trade.py` executes through delegated app `main()` path
- [ ] verify parity for inputs (`mode`, `instrument`, `steps`) between setup config and executed trade run

## Legacy Removal
- [x] remove legacy `client/` folder (migrated to `src/client/`)
- [x] remove legacy `dataloader/` folder (migrated to `src/client/`)
- [x] remove legacy `executor/` folder (migrated to `src/strategy/` and `src/app/`)
- [x] remove legacy `metric/` folder (migrated to `src/measurement/`)
- [x] remove legacy `monitor/` folder (migrated to `src/app/`)
- [x] remove legacy `signal/` folder (migrated to `src/indicator/`)
- [x] remove legacy `strategy/` folder (migrated to `src/strategy/`)
- [x] remove legacy `logger.py` (migrated to `src/app/logger.py`)
- [x] remove legacy `setup.py` (config now in `setup.yaml` and `src/app/config.py`)
- [x] remove legacy `setup_example.py` (config now in `setup.yaml`)
- [x] verify `python backtest.py` still works after legacy removal
- [ ] verify `python trade.py` still works after legacy removal
