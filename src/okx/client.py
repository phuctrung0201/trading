from src.asset.repo import AssetRepo
from src.instrument.repo import InstrumentRepo
from src.okx.auth import OkxAuth
from src.okx.pool import OkxClientPool
from src.trade.repo import TradeRepo


class OkxClient:
    """Facade that composes auth, repos, and a connection pool
    for concurrent rate-limited requests."""

    def __init__(self, api_key: str, secret_key: str, passphrase: str, demo: bool):
        self.auth = OkxAuth(api_key, secret_key, passphrase, demo)
        self.pool = OkxClientPool(api_key, secret_key, passphrase, demo)

        self.instruments = InstrumentRepo(self.pool)
        self.trades = TradeRepo(self.pool, self.auth)
        self.assets = AssetRepo(self.auth)

    def set_api_callback(self, callback):
        self.auth.set_api_callback(callback)
        self.pool.set_api_callback(callback)
