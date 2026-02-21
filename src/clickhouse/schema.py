import pyarrow as pa


_TRADE_EVENT_COLUMNS = [
    ("timestamp", "DateTime64(9, 'UTC')", pa.timestamp("ns", tz="UTC")),
    ("session_id", "String", pa.string()),
    ("setup", "String", pa.string()),
    ("instrument", "String", pa.string()),
    ("strategy", "String", pa.string()),
    ("indicator", "Nullable(String)", pa.string()),
    ("event", "Nullable(String)", pa.string()),
    ("equity", "Float64", pa.float64()),
    ("close_price", "Float64", pa.float64()),
    ("position_size", "Float64", pa.float64()),
    ("position_side", "String", pa.string()),
    ("short_ema", "Nullable(Float64)", pa.float64()),
    ("long_ema", "Nullable(Float64)", pa.float64()),
    ("drawdown", "Float64", pa.float64()),
    ("sharpe_ratio", "Float64", pa.float64()),
    ("exposure_ratio", "Float64", pa.float64()),
    ("fill_price", "Nullable(Float64)", pa.float64()),
    ("pnl", "Nullable(Float64)", pa.float64()),
    ("signal", "Nullable(String)", pa.string()),
    ("reason", "Nullable(String)", pa.string()),
]

_OPS_COLUMNS = [
    ("timestamp", "DateTime64(9, 'UTC')", pa.timestamp("ns", tz="UTC")),
    ("session_id", "String", pa.string()),
    ("setup", "String", pa.string()),
    ("instrument", "String", pa.string()),
    ("strategy", "String", pa.string()),
    ("indicator", "Nullable(String)", pa.string()),
    ("type", "String", pa.string()),
    ("candle_lag_ms", "Nullable(Int64)", pa.int64()),
    ("write_buffer_size", "Nullable(Int64)", pa.int64()),
    ("api_latency_ms", "Nullable(Int64)", pa.int64()),
    ("response_code", "Nullable(Int32)", pa.int32()),
    ("response_source", "Nullable(String)", pa.string()),
    ("reconcile_equity_diff", "Nullable(Float64)", pa.float64()),
    ("reconcile_position_match", "Nullable(UInt8)", pa.uint8()),
    ("reconcile_correction", "Nullable(String)", pa.string()),
    ("error_message", "Nullable(String)", pa.string()),
]

TABLES: dict[str, dict] = {
    "trade_event": {
        "columns": _TRADE_EVENT_COLUMNS,
        "order_by": "(session_id, timestamp)",
    },
    "ops": {
        "columns": _OPS_COLUMNS,
        "order_by": "(session_id, timestamp)",
    },
}


def _arrow_schema(columns: list[tuple]) -> pa.Schema:
    return pa.schema([(name, arrow_type) for name, _, arrow_type in columns])


def _create_ddl(database: str, table: str, spec: dict) -> str:
    col_defs = ",\n    ".join(
        f"{name} {ch_type}" for name, ch_type, _ in spec["columns"]
    )
    return (
        f"CREATE TABLE IF NOT EXISTS {database}.{table} (\n"
        f"    {col_defs}\n"
        f") ENGINE = MergeTree()\n"
        f"PARTITION BY toDate(timestamp)\n"
        f"ORDER BY {spec['order_by']}\n"
    )


TABLE_SCHEMAS: dict[str, pa.Schema] = {
    name: _arrow_schema(spec["columns"]) for name, spec in TABLES.items()
}
