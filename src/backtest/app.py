from src.app.provider import AppProvider
from src.drawdown.strategy import DrawdownStrategy


class BacktestApp:
    def __init__(self, provider: AppProvider):
        self.logger = provider.logger
        self.exchange = provider.simulator
        self.recorder = provider.recorder
        self.session_id = provider.session_id

        setup = provider.setup
        self.instrument = setup.instrument
        self.depth_sec = setup.depth.total_seconds

        self.strategy = DrawdownStrategy(
            exchange=self.exchange,
            recorder=self.recorder,
            logger=self.logger,
            short_length=setup.crossma.short_length,
            long_length=setup.crossma.long_length,
            window=setup.drawdown.window,
            threshold_scale_map=setup.drawdown.threshold_scale_map,
        )

        self.logger.info(
            f"BacktestApp ready session_id {self.session_id} "
            f"instrument {self.instrument} depth_sec {self.depth_sec}s"
        )
