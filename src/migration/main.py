import argparse
import sys

import requests

from src.app.config import load_config, LoggerConfig, ClickHouseConfig
from src.app.logger import init_logger
from src.clickhouse.schema import TABLES, _create_ddl


def _connect(cfg: ClickHouseConfig) -> requests.Session:
    session = requests.Session()
    resp = session.post(
        cfg.url,
        params={"user": cfg.user, "password": cfg.password},
        data="SELECT 1",
        timeout=5,
    )
    resp.raise_for_status()
    return session


def _exec(session: requests.Session, cfg: ClickHouseConfig, query: str):
    resp = session.post(
        cfg.url,
        params={"database": cfg.database, "user": cfg.user, "password": cfg.password},
        data=query,
        timeout=10,
    )
    resp.raise_for_status()
    return resp


def _remote_columns(session: requests.Session, cfg: ClickHouseConfig,
                    table: str) -> dict[str, str]:
    """Return {column_name: column_type} for an existing table."""
    query = (f"SELECT name, type FROM system.columns "
             f"WHERE database = '{cfg.database}' AND table = '{table}'")
    resp = _exec(session, cfg, query)
    result: dict[str, str] = {}
    for line in resp.text.strip().splitlines():
        if not line.strip():
            continue
        name, col_type = line.split("\t", 1)
        result[name.strip()] = col_type.strip()
    return result


def _desired_columns(spec: dict) -> dict[str, str]:
    return {name: ch_type for name, ch_type, _ in spec["columns"]}


def ensure_database(session: requests.Session, cfg: ClickHouseConfig, logger):
    _exec(session, cfg, f"CREATE DATABASE IF NOT EXISTS {cfg.database}")
    logger.info(f"Database ensured: {cfg.database}")


def ensure_tables(session: requests.Session, cfg: ClickHouseConfig, logger):
    for table, spec in TABLES.items():
        _exec(session, cfg, _create_ddl(cfg.database, table, spec))
        logger.info(f"Table ensured: {table}")


def ensure_columns(session: requests.Session, cfg: ClickHouseConfig, logger):
    for table, spec in TABLES.items():
        remote = _remote_columns(session, cfg, table)
        desired = _desired_columns(spec)

        for col_name, ch_type in desired.items():
            if col_name not in remote:
                alter = (f"ALTER TABLE {cfg.database}.{table} "
                         f"ADD COLUMN IF NOT EXISTS {col_name} {ch_type}")
                _exec(session, cfg, alter)
                logger.info(f"Added column {table}.{col_name} {ch_type}")
            elif remote[col_name] != ch_type:
                alter = (f"ALTER TABLE {cfg.database}.{table} "
                         f"MODIFY COLUMN {col_name} {ch_type}")
                _exec(session, cfg, alter)
                logger.info(
                    f"Modified column {table}.{col_name} "
                    f"{remote[col_name]} -> {ch_type}"
                )

        stale = set(remote) - set(desired)
        if stale:
            logger.warning(
                f"Table {table} has columns not in schema: {sorted(stale)}"
            )


def show_ddl(cfg: ClickHouseConfig):
    for table, spec in TABLES.items():
        print(_create_ddl(cfg.database, table, spec))


def parse_args():
    parser = argparse.ArgumentParser(description="Run ClickHouse DDL migrations")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print DDL statements without executing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app_cfg = load_config("app", "logger", LoggerConfig)
    ch_cfg = load_config("clickhouse", "client", ClickHouseConfig)
    logger = init_logger(app_cfg.log_level)

    if args.dry_run:
        show_ddl(ch_cfg)
        return

    if not ch_cfg.enabled:
        logger.error("ClickHouse is not enabled in config")
        sys.exit(1)

    session = _connect(ch_cfg)
    try:
        ensure_database(session, ch_cfg, logger)
        ensure_tables(session, ch_cfg, logger)
        ensure_columns(session, ch_cfg, logger)
        logger.info("Migration complete")
    finally:
        session.close()


if __name__ == "__main__":
    main()
