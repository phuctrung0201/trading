import pyarrow as pa

TRADE_EVENT_DDL = """
CREATE TABLE IF NOT EXISTS {database}.trade_event (
    timestamp DateTime64(9, 'UTC'),
    session_id String,
    setup String,
    instrument String,
    strategy String,
    event Nullable(String),
    equity Float64,
    close_price Float64,
    position_size Float64,
    position_side String,
    short_ema Nullable(Float64),
    long_ema Nullable(Float64),
    drawdown Float64,
    sharpe_ratio Float64,
    exposure_ratio Float64,
    fill_price Nullable(Float64),
    pnl Nullable(Float64),
    signal Nullable(String),
    reason Nullable(String)
) ENGINE = MergeTree()
PARTITION BY toDate(timestamp)
ORDER BY (session_id, timestamp)
"""

OPS_DDL = """
CREATE TABLE IF NOT EXISTS {database}.ops (
    timestamp DateTime64(9, 'UTC'),
    session_id String,
    setup String,
    instrument String,
    strategy String,
    type String,
    candle_lag_ms Nullable(Int64),
    write_buffer_size Nullable(Int64),
    api_latency_ms Nullable(Int64),
    response_code Nullable(Int32),
    response_source Nullable(String),
    reconcile_equity_diff Nullable(Float64),
    reconcile_position_match Nullable(UInt8),
    reconcile_correction Nullable(String),
    error_message Nullable(String)
) ENGINE = MergeTree()
PARTITION BY toDate(timestamp)
ORDER BY (session_id, timestamp)
"""

TRADE_EVENT_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("session_id", pa.string()),
    ("setup", pa.string()),
    ("instrument", pa.string()),
    ("strategy", pa.string()),
    ("event", pa.string()),
    ("equity", pa.float64()),
    ("close_price", pa.float64()),
    ("position_size", pa.float64()),
    ("position_side", pa.string()),
    ("short_ema", pa.float64()),
    ("long_ema", pa.float64()),
    ("drawdown", pa.float64()),
    ("sharpe_ratio", pa.float64()),
    ("exposure_ratio", pa.float64()),
    ("fill_price", pa.float64()),
    ("pnl", pa.float64()),
    ("signal", pa.string()),
    ("reason", pa.string()),
])

OPS_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("session_id", pa.string()),
    ("setup", pa.string()),
    ("instrument", pa.string()),
    ("strategy", pa.string()),
    ("type", pa.string()),
    ("candle_lag_ms", pa.int64()),
    ("write_buffer_size", pa.int64()),
    ("api_latency_ms", pa.int64()),
    ("response_code", pa.int32()),
    ("response_source", pa.string()),
    ("reconcile_equity_diff", pa.float64()),
    ("reconcile_position_match", pa.uint8()),
    ("reconcile_correction", pa.string()),
    ("error_message", pa.string()),
])

TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "trade_event": TRADE_EVENT_SCHEMA,
    "ops": OPS_SCHEMA,
}
