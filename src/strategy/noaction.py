from src.app.logger import AppLogger
from src.client.ohclv import OHCLV


class NoActionStrategy:
    def __init__(self, app_logger: AppLogger):
        self._logger = app_logger

    def ack(self, candle: OHCLV):
        timestamp = getattr(candle, "timestamp", None)
        self._logger.info(f"NoActionStrategy ack timestamp={timestamp}")
        return None
