"""Centralised logger with level configured via setup.log_level.

Usage
-----
::

    from logger import log

    log.info("Connected")
    log.debug("tick data: ...")
    log.warn("Drawdown exceeded threshold")
    log.error("Order failed")

Log levels (lowest → highest):
    DEBUG < INFO < WARN < ERROR < SILENT

Set in setup.py::

    log_level = "INFO"   # or "DEBUG", "WARN", "ERROR", "SILENT"
"""

from __future__ import annotations

from datetime import datetime, timezone

_LEVELS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARN": 2,
    "ERROR": 3,
    "SILENT": 4,
}


class Logger:
    """Simple level-filtered logger that writes to stdout."""

    def __init__(self, level: str = "INFO") -> None:
        self.set_level(level)

    def set_level(self, level: str) -> None:
        level = level.upper()
        if level not in _LEVELS:
            raise ValueError(
                f"Invalid log level '{level}'. "
                f"Choose from: {', '.join(_LEVELS)}"
            )
        self._level = _LEVELS[level]
        self._level_name = level

    def _log(self, tag: str, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"{ts} [{tag}] {msg}")

    def debug(self, msg: str) -> None:
        if self._level <= _LEVELS["DEBUG"]:
            self._log("DEBUG", msg)

    def info(self, msg: str) -> None:
        if self._level <= _LEVELS["INFO"]:
            self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        if self._level <= _LEVELS["WARN"]:
            self._log("WARN", msg)

    def error(self, msg: str) -> None:
        if self._level <= _LEVELS["ERROR"]:
            self._log("ERROR", msg)


# Singleton – import and use directly: from logger import log
log = Logger("INFO")


def configure(level: str) -> None:
    """Reconfigure the global log level."""
    log.set_level(level)
