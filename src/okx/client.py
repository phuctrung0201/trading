from src.okx.auth import OkxAuth
from src.okx.pool import OkxClientPool
from src.okx.trading import OkxTrading
from src.okx.account import OkxAccount


class OkxClient:
    """Facade that composes auth, trading, account sub-clients, and a
    connection pool for concurrent rate-limited requests."""

    def __init__(self, api_key: str, secret_key: str, passphrase: str, demo: bool):
        self.auth = OkxAuth(api_key, secret_key, passphrase, demo)
        self.trading = OkxTrading(self.auth)
        self.account = OkxAccount(self.auth)
        self.pool = OkxClientPool(api_key, secret_key, passphrase, demo)

    def set_api_callback(self, callback):
        self.auth.set_api_callback(callback)
        self.pool.set_api_callback(callback)
