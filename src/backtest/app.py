from src.app.provider import AppProvider


class BacktestApp:
    def __init__(self, provider: AppProvider):
        self.logger = provider.logger
        self.session_id = provider.session_id
        self.exchange = provider.simulator
        self.strategy = provider.strategy
        self.clickhouse_client = provider.clickhouse_client
