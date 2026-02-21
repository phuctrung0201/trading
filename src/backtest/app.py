from src.app.provider import AppProvider
from src.drawdown.strategy import DrawdownStrategy


class BacktestApp:
    def __init__(self, provider: AppProvider):
        self.logger = provider.logger
        self.exchange = provider.simulator
        self.recorder = provider.recorder
        self.okx_client = provider.okx_client
        self.session_id = provider.session_id

        setup = provider.setup
        exchange_cfg = setup["exchange"]
        self.instrument = exchange_cfg["instrument"]
        self.step = exchange_cfg.get("steps", "1m")
        self.backtest_start = exchange_cfg["start"]
        self.backtest_end = exchange_cfg["end"]

        crossma_cfg = setup.get("crossma", {})
        self.warmup_periods = int(crossma_cfg.get("long_length", 200))

        drawdown_cfg = setup.get("drawdown", {})
        self.strategy = DrawdownStrategy(
            exchange=self.exchange,
            recorder=self.recorder,
            logger=self.logger,
            short_length=int(crossma_cfg.get("short_length", 15)),
            long_length=int(crossma_cfg.get("long_length", 200)),
            steps=self.step,
            window=int(drawdown_cfg.get("window", 500)),
            threshold_scale_map=drawdown_cfg.get("threshold_scale_map", {0.0: 1.0}),
        )

        self.logger.info(
            f"BacktestApp ready instrument={self.instrument} step={self.step} "
            f"start={self.backtest_start} end={self.backtest_end}"
        )
