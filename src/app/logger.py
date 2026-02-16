import logging
import sys


class AppLogger:
    def __init__(self, logger):
        self._logger = logger

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)

    def warn(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)


def init_logger(log_level):
    logger = logging.getLogger("trading.app")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    return AppLogger(logger)
