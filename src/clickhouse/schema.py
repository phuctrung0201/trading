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
    ("price", "Float64", pa.float64()),
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
    ("zscore", "Nullable(Float64)", pa.float64()),
    ("fee", "Nullable(Float64)", pa.float64()),
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

_MARKET_TRADE_COLUMNS = [
    ("dataset", "String", pa.string()),
    ("trade_id", "String", pa.string()),
    ("timestamp", "DateTime64(3, 'UTC')", pa.timestamp("ms", tz="UTC")),
    ("price", "Float64", pa.float64()),
    ("size", "Float64", pa.float64()),
    ("side", "String", pa.string()),
]

_PORTFOLIO_SESSION_COLUMNS = [
    ("session_id", "String", pa.string()),
    ("started_at", "DateTime64(3, 'UTC')", pa.timestamp("ms", tz="UTC")),
    ("finished_at", "Nullable(DateTime64(3, 'UTC'))", pa.timestamp("ms", tz="UTC")),
    ("status", "String", pa.string()),
    ("config_name", "String", pa.string()),
    ("universe_size", "UInt32", pa.uint32()),
    ("passed_count", "UInt32", pa.uint32()),
    ("error_message", "Nullable(String)", pa.string()),
]

_PORTFOLIO_SCREEN_COLUMNS = [
    ("session_id", "String", pa.string()),
    ("instrument", "String", pa.string()),
    ("adf_stat", "Nullable(Float64)", pa.float64()),
    ("adf_pvalue", "Nullable(Float64)", pa.float64()),
    ("hurst", "Nullable(Float64)", pa.float64()),
    ("half_life", "Nullable(Float64)", pa.float64()),
    ("volatility", "Nullable(Float64)", pa.float64()),
    ("daily_volume", "Float64", pa.float64()),
    ("passed", "UInt8", pa.uint8()),
    ("fail_reason", "Nullable(String)", pa.string()),
]

_PORTFOLIO_RANKING_COLUMNS = [
    ("session_id", "String", pa.string()),
    ("instrument", "String", pa.string()),
    ("rank", "UInt32", pa.uint32()),
    ("composite_score", "Float64", pa.float64()),
    ("adf_pvalue", "Float64", pa.float64()),
    ("hurst", "Float64", pa.float64()),
    ("half_life", "Float64", pa.float64()),
    ("volatility", "Float64", pa.float64()),
]

_FUNDING_SESSION_COLUMNS = [
    ("session_id", "String", pa.string()),
    ("timestamp", "DateTime64(3, 'UTC')", pa.timestamp("ms", tz="UTC")),
    ("status", "String", pa.string()),
    ("error_message", "Nullable(String)", pa.string()),
]

_FUNDING_SCREEN_COLUMNS = [
    ("session_id", "String", pa.string()),
    ("pair", "String", pa.string()),
    ("inst_id", "String", pa.string()),
    ("direction", "String", pa.string()),
    ("funding_rate", "Float64", pa.float64()),
    ("timestamp", "DateTime64(3, 'UTC')", pa.timestamp("ms", tz="UTC")),
]

_FUNDING_MONITOR_COLUMNS = [
    ("pair", "String", pa.string()),
    ("direction", "String", pa.string()),
    ("spot_notional", "Float64", pa.float64()),
    ("perp_notional", "Float64", pa.float64()),
    ("drift", "Float64", pa.float64()),
    ("current_funding_rate", "Float64", pa.float64()),
    ("timestamp", "DateTime64(3, 'UTC')", pa.timestamp("ms", tz="UTC")),
]

TABLES: dict[str, dict] = {
    "trade_event": {
        "columns": _TRADE_EVENT_COLUMNS,
        "order_by": "(session_id, timestamp)",
        "partition_by": "toDate(timestamp)",
    },
    "ops": {
        "columns": _OPS_COLUMNS,
        "order_by": "(session_id, timestamp)",
        "partition_by": "toDate(timestamp)",
    },
    "market_trade": {
        "columns": _MARKET_TRADE_COLUMNS,
        "order_by": "(dataset, timestamp)",
        "partition_by": "toDate(timestamp)",
    },
    "portfolio_session": {
        "columns": _PORTFOLIO_SESSION_COLUMNS,
        "order_by": "(session_id)",
        "engine": "ReplacingMergeTree(started_at)",
    },
    "portfolio_screen": {
        "columns": _PORTFOLIO_SCREEN_COLUMNS,
        "order_by": "(session_id, instrument)",
    },
    "portfolio_ranking": {
        "columns": _PORTFOLIO_RANKING_COLUMNS,
        "order_by": "(session_id, rank)",
    },
    "funding_session": {
        "columns": _FUNDING_SESSION_COLUMNS,
        "order_by": "(session_id)",
        "engine": "ReplacingMergeTree(timestamp)",
    },
    "funding_screen": {
        "columns": _FUNDING_SCREEN_COLUMNS,
        "order_by": "(session_id, pair)",
        "partition_by": "toDate(timestamp)",
    },
    "funding_monitor": {
        "columns": _FUNDING_MONITOR_COLUMNS,
        "order_by": "(pair, timestamp)",
        "partition_by": "toDate(timestamp)",
    },
}


def _arrow_schema(columns: list[tuple]) -> pa.Schema:
    return pa.schema([(name, arrow_type) for name, _, arrow_type in columns])


def _create_ddl(database: str, table: str, spec: dict) -> str:
    col_defs = ",\n    ".join(
        f"{name} {ch_type}" for name, ch_type, _ in spec["columns"]
    )
    engine = spec.get("engine", "MergeTree()")
    partition = spec.get("partition_by")
    partition_clause = f"PARTITION BY {partition}\n" if partition else ""
    return (
        f"CREATE TABLE IF NOT EXISTS {database}.{table} (\n"
        f"    {col_defs}\n"
        f") ENGINE = {engine}\n"
        f"{partition_clause}"
        f"ORDER BY {spec['order_by']}\n"
    )


TABLE_SCHEMAS: dict[str, pa.Schema] = {
    name: _arrow_schema(spec["columns"]) for name, spec in TABLES.items()
}
