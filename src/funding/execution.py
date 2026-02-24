from __future__ import annotations

import time
from dataclasses import dataclass

from src.app.config import FundingConfig
from src.app.logger import AppLogger
from src.funding.data import get_funding_rate, get_prices, PricePair
from src.okx.pool import OkxClientPool
from src.okx.trading import OkxTrading


@dataclass
class FundingPosition:
    pair: str
    spot_inst_id: str
    perp_inst_id: str
    direction: str
    notional: float
    spot_qty: float
    perp_qty: float
    spot_entry_price: float
    perp_entry_price: float
    entry_funding_rate: float
    min_funding_rate: float


def _spot_inst_id(pair: str, quote: str) -> str:
    return f"{pair}-{quote}"


def _perp_inst_id(pair: str, quote: str) -> str:
    return f"{pair}-{quote}-SWAP"


def enter_position(
    pool: OkxClientPool,
    trading: OkxTrading,
    pair: str,
    direction: str,
    config: FundingConfig,
    logger: AppLogger,
) -> FundingPosition | None:
    """Open a delta-neutral spot+perp position.

    direction: "long_spot" (long spot + short perp) or "short_spot".
    """
    quote = config.quote
    spot_id = _spot_inst_id(pair, quote)
    perp_id = _perp_inst_id(pair, quote)
    notional = config.notional

    prices = get_prices(pool, spot_id, perp_id)
    spot_qty = notional / prices.spot_price
    perp_qty = notional / prices.perp_price

    if direction == "long_spot":
        spot_side, perp_side = "buy", "sell"
    else:
        spot_side, perp_side = "sell", "buy"

    spot_order = _place_with_retry(
        trading, spot_id, spot_side, spot_qty, config.retry_count, logger,
    )
    if spot_order is None:
        logger.error(f"enter_position: spot leg failed pair={pair}")
        return None

    perp_order = _place_with_retry(
        trading, perp_id, perp_side, perp_qty, config.retry_count, logger,
    )
    if perp_order is None:
        logger.error(f"enter_position: perp leg failed, flattening spot pair={pair}")
        _flatten_leg(trading, spot_id, "sell" if spot_side == "buy" else "buy", spot_qty, logger)
        return None

    current_rate = get_funding_rate(pool, perp_id)

    return FundingPosition(
        pair=pair,
        spot_inst_id=spot_id,
        perp_inst_id=perp_id,
        direction=direction,
        notional=notional,
        spot_qty=spot_qty,
        perp_qty=perp_qty,
        spot_entry_price=prices.spot_price,
        perp_entry_price=prices.perp_price,
        entry_funding_rate=current_rate.rate,
        min_funding_rate=config.min_funding_rate,
    )


def rebalance(
    pool: OkxClientPool,
    trading: OkxTrading,
    position: FundingPosition,
    config: FundingConfig,
    logger: AppLogger,
) -> FundingPosition:
    """Check drift and rebalance if needed. Returns updated position."""
    prices = get_prices(pool, position.spot_inst_id, position.perp_inst_id)
    spot_notional = position.spot_qty * prices.spot_price
    perp_notional = position.perp_qty * prices.perp_price

    mid_notional = (spot_notional + perp_notional) / 2
    drift = abs(spot_notional - perp_notional)
    band = config.drift_band * mid_notional

    if drift <= band:
        logger.info(
            f"rebalance: pair={position.pair} drift={drift:.2f} band={band:.2f} — hold"
        )
        return position

    logger.info(
        f"rebalance: pair={position.pair} drift={drift:.2f} band={band:.2f} — rebalancing"
    )

    target_notional = mid_notional
    new_spot_qty = target_notional / prices.spot_price
    new_perp_qty = target_notional / prices.perp_price

    spot_diff = new_spot_qty - position.spot_qty
    perp_diff = new_perp_qty - position.perp_qty

    if abs(spot_diff) > 1e-12:
        side = "buy" if spot_diff > 0 else "sell"
        _place_with_retry(
            trading, position.spot_inst_id, side, abs(spot_diff),
            config.retry_count, logger,
        )

    if abs(perp_diff) > 1e-12:
        if position.direction == "long_spot":
            side = "sell" if perp_diff > 0 else "buy"
        else:
            side = "buy" if perp_diff > 0 else "sell"
        _place_with_retry(
            trading, position.perp_inst_id, side, abs(perp_diff),
            config.retry_count, logger,
        )

    position.spot_qty = new_spot_qty
    position.perp_qty = new_perp_qty
    position.notional = target_notional
    return position


def check_exit(
    pool: OkxClientPool,
    position: FundingPosition,
    logger: AppLogger,
) -> bool:
    """Return True if the position should be exited."""
    rate_data = get_funding_rate(pool, position.perp_inst_id)
    current_rate = rate_data.rate

    if position.direction == "long_spot":
        flipped = current_rate < 0
    else:
        flipped = current_rate > 0

    collapsed = abs(current_rate) < position.min_funding_rate

    if flipped:
        logger.info(f"check_exit: pair={position.pair} rate flipped ({current_rate:.6f})")
        return True
    if collapsed:
        logger.info(
            f"check_exit: pair={position.pair} rate collapsed "
            f"({current_rate:.6f} < {position.min_funding_rate:.6f})"
        )
        return True

    logger.info(f"check_exit: pair={position.pair} rate={current_rate:.6f} — hold")
    return False


def exit_position(
    trading: OkxTrading,
    position: FundingPosition,
    logger: AppLogger,
) -> None:
    """Close both legs of the position."""
    if position.direction == "long_spot":
        spot_close_side, perp_close_side = "sell", "buy"
    else:
        spot_close_side, perp_close_side = "buy", "sell"

    logger.info(f"exit_position: closing pair={position.pair}")
    _flatten_leg(trading, position.spot_inst_id, spot_close_side, position.spot_qty, logger)
    _flatten_leg(trading, position.perp_inst_id, perp_close_side, position.perp_qty, logger)


def _place_with_retry(
    trading: OkxTrading,
    inst_id: str,
    side: str,
    size: float,
    retries: int,
    logger: AppLogger,
) -> list | None:
    for attempt in range(1, retries + 1):
        try:
            result = trading.place_order(inst_id, side, size)
            return result
        except Exception as exc:
            logger.warning(
                f"Order failed inst={inst_id} side={side} size={size:.6f} "
                f"attempt={attempt}/{retries}: {exc}"
            )
            if attempt < retries:
                time.sleep(1.0)
    return None


def _flatten_leg(
    trading: OkxTrading,
    inst_id: str,
    side: str,
    size: float,
    logger: AppLogger,
) -> None:
    try:
        trading.place_order(inst_id, side, size)
    except Exception as exc:
        logger.error(f"Flatten failed inst={inst_id} side={side}: {exc}")
