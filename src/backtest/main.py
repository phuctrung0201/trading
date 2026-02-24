import argparse
from collections.abc import Iterator

from src.app.config import load_config, StrategyConfig
from src.app.provider import AppProvider
from src.backtest.app import BacktestApp
from src.exchange.dto import FundingSnapshot
from src.exchange.simulator import SimulateExchange
from src.universe import Universe


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--setup", type=str, required=True, help="Setup name from config/strategy/")
    return parser.parse_args()


def _build_trade_source(provider: AppProvider, setup: StrategyConfig):
    bt = setup.backtest
    dataset = bt.dataset
    if dataset:
        from src.clickhouse.dataset import DatasetReader
        if provider.clickhouse_client is None:
            raise RuntimeError("ClickHouse must be enabled for dataset backtest")
        reader = DatasetReader(provider.clickhouse_client, dataset, provider.logger)
        return reader.stream()
    depth_sec = bt.depth.total_minutes * 60 if bt.depth else None
    return provider.okx_exchange.stream_history(depth_sec=depth_sec)


def _fetch_candle_close(pool, inst_id: str, ts: int) -> float:
    """Fetch the 4H candle close price at the given timestamp.

    Tries the recent candles endpoint first, then falls back to
    history-candles for older data.
    """
    params = {"instId": inst_id, "bar": "4H", "after": str(ts + 1), "limit": "1"}
    try:
        candles = pool.public_get("/api/v5/market/candles", params=params)
        if candles:
            return float(candles[0][4])
        candles = pool.public_get("/api/v5/market/history-candles", params=params)
        if candles:
            return float(candles[0][4])
    except Exception:
        pass
    return 0.0


def _fetch_instrument_funding(
    pool, pair: str, perp_id: str, logger,
) -> Iterator[FundingSnapshot]:
    """Yield historical funding snapshots for a single instrument."""
    quote = perp_id.split("-")[1] if "-" in perp_id else "USDT"
    spot_id = f"{pair}-{quote}"

    history = pool.public_get(
        "/api/v5/public/funding-rate-history",
        params={"instId": perp_id, "limit": "100"},
    )
    if not history:
        logger.warning(f"No funding history for {perp_id}")
        return

    history.sort(key=lambda r: int(r.get("fundingTime", 0)))

    for entry in history:
        rate = float(entry.get("realizedRate", entry.get("fundingRate", "0")))
        ts = int(entry.get("fundingTime", 0))

        spot_price = _fetch_candle_close(pool, spot_id, ts)
        perp_price = _fetch_candle_close(pool, perp_id, ts)

        if spot_price <= 0 or perp_price <= 0:
            continue

        yield FundingSnapshot(
            timestamp=ts,
            funding_rate=rate,
            spot_price=spot_price,
            perp_price=perp_price,
            inst_id=perp_id,
        )


def _build_funding_source(
    provider: AppProvider, universe: Universe,
) -> Iterator[FundingSnapshot]:
    """Fetch historical funding snapshots for all instruments in the universe,
    merged in chronological order."""
    pool = provider.okx_client.pool
    logger = provider.logger

    logger.info(f"Fetching funding history for {len(universe)} instruments")

    all_snapshots: list[FundingSnapshot] = []
    for inst in universe:
        snapshots = list(_fetch_instrument_funding(pool, inst.pair, inst.inst_id, logger))
        logger.info(f"  {inst.inst_id}: {len(snapshots)} snapshots")
        all_snapshots.extend(snapshots)

    all_snapshots.sort(key=lambda s: (s.timestamp, s.inst_id))
    yield from all_snapshots


def _run_funding(app: BacktestApp, provider: AppProvider, universe: Universe):
    total = 0
    for snapshot in _build_funding_source(provider, universe):
        total += 1
        try:
            app.strategy.ack_funding(snapshot)
        except Exception as exc:
            app.logger.error(
                f"Strategy ack_funding failed inst={snapshot.inst_id} "
                f"timestamp={snapshot.timestamp} "
                f"rate={snapshot.funding_rate} total_processed={total}: {exc}",
                exc_info=True,
            )
            raise
    return total


def _run_trade(app: BacktestApp, provider: AppProvider, setup: StrategyConfig):
    total = 0
    for trade in _build_trade_source(provider, setup):
        total += 1
        try:
            app.exchange.set_price(trade.price)
            app.strategy.ack_trade(trade)
        except Exception as exc:
            app.logger.error(
                f"Strategy ack_trade failed timestamp={trade.timestamp} "
                f"price={trade.price} total_processed={total}: {exc}",
                exc_info=True,
            )
            raise
    return total


def main():
    args = parse_args()
    provider = AppProvider()
    setup = load_config("strategy", args.setup, StrategyConfig)

    is_funding = setup.data_source == "funding"

    universe: Universe | None = None
    if setup.universe:
        universe = provider.universe.discover(setup.universe)

    provider.simulator = SimulateExchange(
        initial_equity=setup.funding.notional if is_funding else setup.backtest.initial_equity,
        fee_rate=0.0 if is_funding else setup.backtest.fee_rate,
    )

    instrument_label = ",".join(universe.inst_ids) if universe else setup.instrument
    provider.recorder.bootstrap(
        setup_name=args.setup, instrument=instrument_label,
        strategy=setup.strategy,
    )

    strategy = provider.strategy_factory.build(
        setup, exchange=provider.simulator, universe=universe,
    )
    app = BacktestApp(provider, strategy, provider.simulator)
    logger = app.logger

    try:
        if is_funding:
            total = _run_funding(app, provider, universe)
        else:
            total = _run_trade(app, provider, setup)

        logger.info(f"Backtest completed total_steps={total}")
        if total == 0:
            logger.warning("No data returned; strategy ack did not run")

        logger.info(f"Backtest session_id {app.session_id}")
        logger.info(f"Final equity={app.exchange.get_equity():.4f}")

        if is_funding:
            res = strategy.results()
            logger.info(f"  Funding earned={res.funding_earned:.4f}")
            logger.info(f"  Rebalances={res.rebalance_count}")
            logger.info(f"  Max drawdown={res.max_drawdown:.4%}")
            logger.info(f"  Positions opened={res.positions_opened} closed={res.positions_closed}")
        else:
            logger.info(f"Total fees={app.exchange.total_fees:.4f}")
    finally:
        if provider.clickhouse_client is not None:
            provider.clickhouse_client.close()


if __name__ == "__main__":
    main()
