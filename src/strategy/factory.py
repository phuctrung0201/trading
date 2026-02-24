from src.app.config import StrategyConfig
from src.app.logger import AppLogger
from src.clickhouse.recorder import Recorder
from src.drawdown.meanrev import DrawdownMeanRevStrategy
from src.drawdown.strategy import DrawdownCrossMAStrategy
from src.exchange.adapter import ExchangeAdapter
from src.funding.strategy import FundingStrategy
from src.universe import Universe
from src.strategy.adapter import StrategyAdapter


class StrategyFactory:
    def __init__(self, recorder: Recorder, logger: AppLogger):
        self._recorder = recorder
        self._logger = logger

    def build(
        self,
        setup: StrategyConfig,
        exchange: ExchangeAdapter,
        universe: Universe | None = None,
    ) -> StrategyAdapter:
        name = setup.strategy

        if name == "funding":
            strategy = FundingStrategy(
                recorder=self._recorder, logger=self._logger,
            )
            strategy.bootstrap(
                exchange=exchange, config=setup.funding,
                universe=universe or Universe([]),
            )
            return strategy

        if name == "drawdown_meanrev":
            strategy = DrawdownMeanRevStrategy(
                recorder=self._recorder, logger=self._logger,
            )
            strategy.bootstrap(
                exchange=exchange,
                bucket_interval=setup.meanrev.bucket_interval,
                window=setup.drawdown.window,
                threshold_scale_map=setup.drawdown.threshold_scale_map,
                lookback=setup.meanrev.lookback,
                entry_threshold=setup.meanrev.entry_threshold,
                exit_threshold=setup.meanrev.exit_threshold,
            )
            return strategy

        if name in ("drawdown_crossma", "drawdown"):
            strategy = DrawdownCrossMAStrategy(
                recorder=self._recorder, logger=self._logger,
            )
            strategy.bootstrap(
                exchange=exchange,
                short_length=setup.crossma.short_length,
                long_length=setup.crossma.long_length,
                bucket_interval=setup.crossma.bucket_interval,
                window=setup.drawdown.window,
                threshold_scale_map=setup.drawdown.threshold_scale_map,
            )
            return strategy

        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: funding, drawdown_meanrev, drawdown_crossma"
        )
