from src.app.provider import AppProvider


class BacktestApp:
    def __init__(self, provider: AppProvider):
        self.logger = provider.logger
        self.exchange = provider.exchange
        self.strategy = provider.strategy
        self.session_id = provider.session_id
        self.depth_sec = provider.setup.depth.total_minutes * 60

        self.logger.info(
            f"BacktestApp ready session_id {self.session_id} "
            f"instrument {provider.setup.instrument} depth_sec {self.depth_sec}s"
        )
