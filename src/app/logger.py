import logging
import sys


class AppLogger:
    def __init__(self, logger):
        self._logger = logger

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)

    def warn(self, message, exc_info=False):
        self._logger.warning(message, exc_info=exc_info)

    def warning(self, message, exc_info=False):
        self._logger.warning(message, exc_info=exc_info)

    def error(self, message, exc_info=False):
        self._logger.error(message, exc_info=exc_info)


def init_logger(log_level):
    logger = logging.getLogger("trading.app")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.ERROR)
    logger.addHandler(stderr_handler)

    logger.propagate = False

    return AppLogger(logger)
