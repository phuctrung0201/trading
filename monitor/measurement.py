"""Central measurement names and helpers used by monitoring and ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def _format_fields(fields: dict[str, float | int | str | bool]) -> str:
    parts: list[str] = []
    for key, val in fields.items():
        field_key = _escape(key)
        if isinstance(val, bool):
            field_val = "true" if val else "false"
        elif isinstance(val, int) and not isinstance(val, bool):
            field_val = f"{val}i"
        elif isinstance(val, float):
            field_val = repr(val)
        else:
            escaped = str(val).replace('"', '\\"')
            field_val = f'"{escaped}"'
        parts.append(f"{field_key}={field_val}")
    return ",".join(parts)


class MeasurementValue:
    def __init__(
        self,
        *,
        measurement: str,
        timestamp_ns: int,
        tags: dict[str, str] | None = None,
        **field_values: float | int | str | bool,
    ) -> None:
        self.measurement = measurement
        self.timestamp_ns = timestamp_ns
        self.tags = tags
        self._field_values = field_values

    def to_line(self) -> str:
        tag_segment = ""
        if self.tags:
            sorted_tags = sorted((k, v) for k, v in self.tags.items() if v != "")
            if sorted_tags:
                tag_segment = "," + ",".join(
                    f"{_escape(str(key))}={_escape(str(val))}" for key, val in sorted_tags
                )
        field_segment = _format_fields(self._field_values)
        return f"{_escape(self.measurement)}{tag_segment} {field_segment} {self.timestamp_ns}"


@dataclass(slots=True)
class NoValueMeasurement:
    """Event-style measurement without a business value field."""

    measurement: str
    timestamp_ns: int
    tags: dict[str, str] | None = None


@dataclass(slots=True)
class BacktestMeasurement(NoValueMeasurement):
    """Backtest measurement supporting full field payloads."""

    NAME: ClassVar[str] = "backtest"
    measurement: str

    def __init__(self) -> None:
        self.measurement = self.NAME
        self.tags = None

    def values(
        self,
        *,
        backtest_id: str,
        timestamp_ns: int,
        drawdown: float,
        equity: float,
        sharpe_ratio: float,
    ) -> MeasurementValue:
        values: dict[str, float | int | str | bool] = {
            "backtest_id": backtest_id,
            "drawdown": drawdown,
            "equity": equity,
            "sharpe_ratio": sharpe_ratio,
        }

        return MeasurementValue(
            measurement=self.measurement,
            timestamp_ns=timestamp_ns,
            tags=self.tags,
            **values,
        )


@dataclass(slots=True)
class TradeMeasurement(NoValueMeasurement):
    """Live trade measurement tagged by trading session."""

    NAME: ClassVar[str] = "trade"
    measurement: str

    def __init__(self) -> None:
        self.measurement = self.NAME
        self.tags = None

    def values(
        self,
        *,
        timestamp_ns: int,
        session_id: str,
        equity: float,
        position_side: str,
        position_size: float,
    ) -> MeasurementValue:
        values: dict[str, float | int | str | bool] = {
            "session_id": session_id,
            "equity": equity,
            "position_side": position_side,
            "position_size": position_size,
        }

        return MeasurementValue(
            measurement=self.measurement,
            timestamp_ns=timestamp_ns,
            tags=self.tags,
            **values,
        )


@dataclass(slots=True)
class ClientRequestMeasurement(NoValueMeasurement):
    """Exchange client REST request measurement."""

    NAME: ClassVar[str] = "client_request"
    measurement: str

    def __init__(
        self,
        *,
        client_name: str,
    ) -> None:
        self.measurement = self.NAME
        self.tags = {
            "client_name": client_name,
        }

    def values(
        self,
        *,
        timestamp_ns: int,
        session_id: str,
        status_code: int,
    ) -> MeasurementValue:
        values: dict[str, float | int | str | bool] = {
            "session_id": session_id,
            "status_code": status_code,
        }

        return MeasurementValue(
            measurement=self.measurement,
            timestamp_ns=timestamp_ns,
            tags=self.tags,
            **values,
        )
