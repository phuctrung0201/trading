from src.app.provider import AppProvider
from src.exchange.adapter import ExchangeAdapter


class BacktestApp:
    def __init__(self, provider: AppProvider, strategy, exchange: ExchangeAdapter):
        self.logger = provider.logger
        self.session_id = provider.session_id
        self.exchange = exchange
        self.strategy = strategy
        self.clickhouse_client = provider.clickhouse_client
