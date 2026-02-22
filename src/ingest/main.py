import argparse
from datetime import datetime

from src.app.provider import AppProvider
from src.ingest.app import IngestApp


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest trade data into a dataset")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (e.g. SOL-TRADE-OKX)")
    parser.add_argument("--instrument", type=str, required=True, help="Instrument (e.g. SOL-USDT-SWAP)")
    parser.add_argument("--start", type=str, required=True, help="Start time ISO 8601 (e.g. 2026-01-01T00:00:00Z)")
    parser.add_argument("--end", type=str, required=True, help="End time ISO 8601 (e.g. 2026-02-22T00:00:00Z)")
    return parser.parse_args()


def _iso_to_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def main():
    args = parse_args()
    provider = AppProvider()
    provider.okx_exchange.bootstrap(instrument=args.instrument)
    app = IngestApp(provider)

    provider.logger.info(
        f"Ingest starting instrument={args.instrument} "
        f"dataset={args.name} start={args.start} end={args.end}"
    )

    try:
        app.run(args.name, _iso_to_ms(args.start), _iso_to_ms(args.end))
    finally:
        if provider.clickhouse_client is not None:
            provider.clickhouse_client.close()


if __name__ == "__main__":
    main()
