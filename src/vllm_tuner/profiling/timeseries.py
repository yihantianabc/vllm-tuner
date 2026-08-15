"""Small, dependency-free helpers for telemetry time series.

The telemetry pipeline deliberately stores both a monotonic timestamp (for
durations and cross-layer alignment) and a wall-clock timestamp (for humans).
This module also centralises aggregation semantics so missing telemetry is
represented by ``None`` rather than a misleading zero.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TextIO, Union

Number = Union[int, float]


@dataclass(frozen=True)
class SampleTimestamp:
    """A pair of timestamps captured as closely together as possible."""

    monotonic_ns: int
    wall_time: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"monotonic_ns": self.monotonic_ns, "wall_time": self.wall_time}


def capture_timestamp(
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> SampleTimestamp:
    """Capture monotonic and wall-clock timestamps for one sampling instant."""
    monotonic_value = int(monotonic_ns())
    wall_value = wall_clock()
    if wall_value.tzinfo is None:
        wall_value = wall_value.replace(tzinfo=timezone.utc)
    return SampleTimestamp(
        monotonic_ns=monotonic_value,
        wall_time=wall_value.astimezone(timezone.utc).isoformat(),
    )


def percentile(values: Iterable[Number], quantile: float) -> Optional[float]:
    """Calculate a linearly interpolated percentile.

    ``quantile`` accepts either a fraction in ``[0, 1]`` or a percentile in
    ``[0, 100]``. Non-finite values are ignored and an empty input is explicitly
    unavailable (``None``).
    """
    if quantile > 1.0:
        quantile /= 100.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1 (or 0 and 100)")

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize_values(values: Iterable[Optional[Number]]) -> dict[str, Any]:
    """Return count/mean/peak/p95 for finite values.

    The result shape is stable even when no samples exist. This is important to
    distinguish an actually measured zero from unavailable telemetry.
    """
    available_values = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if not available_values:
        return {
            "available": False,
            "count": 0,
            "mean": None,
            "peak": None,
            "max": None,
            "p95": None,
        }

    peak = max(available_values)
    return {
        "available": True,
        "count": len(available_values),
        "mean": math.fsum(available_values) / len(available_values),
        "peak": peak,
        "max": peak,
        "p95": percentile(available_values, 0.95),
    }


def counter_window_delta(values: Iterable[Optional[Number]]) -> dict[str, Any]:
    """Calculate a counter delta over a measurement window.

    Counter resets are handled by treating the first value after a decrease as
    the post-reset contribution. At least two finite observations are required;
    a single cumulative value cannot describe activity inside the window.
    """
    samples = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if len(samples) < 2:
        return {
            "available": False,
            "sample_count": len(samples),
            "start": samples[0] if samples else None,
            "end": samples[-1] if samples else None,
            "delta": None,
            "reset_count": 0,
        }

    delta = 0.0
    reset_count = 0
    previous = samples[0]
    for current in samples[1:]:
        if current >= previous:
            delta += current - previous
        else:
            reset_count += 1
            delta += current
        previous = current

    return {
        "available": True,
        "sample_count": len(samples),
        "start": samples[0],
        "end": samples[-1],
        "delta": delta,
        "reset_count": reset_count,
    }


def integrate_power_samples(
    samples: Sequence[tuple[int, Optional[Number]]],
) -> Optional[float]:
    """Integrate power samples with the trapezoidal rule and return joules.

    Only adjacent valid samples form an interval. This avoids silently bridging
    a telemetry outage and overestimating energy.
    """
    if len(samples) < 2:
        return None

    energy_joules = 0.0
    interval_count = 0
    for (start_ns, start_power), (end_ns, end_power) in zip(samples, samples[1:]):
        if start_power is None or end_power is None or end_ns <= start_ns:
            continue
        start_value = float(start_power)
        end_value = float(end_power)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            continue
        duration_seconds = (end_ns - start_ns) / 1_000_000_000.0
        energy_joules += (start_value + end_value) * 0.5 * duration_seconds
        interval_count += 1

    return energy_joules if interval_count else None


def make_json_safe(value: Any) -> Any:
    """Recursively convert common telemetry values into JSON-safe objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class JSONLWriter:
    """A tiny synchronous JSONL writer with deterministic cleanup."""

    def __init__(self, path: Union[str, Path], append: bool = False) -> None:
        self.path = Path(path)
        self.append = append
        self._file: Optional[TextIO] = None

    @property
    def is_open(self) -> bool:
        """Whether the output file is currently open."""
        return self._file is not None and not self._file.closed

    def open(self) -> None:
        """Open the writer, creating parent directories when necessary."""
        if self.is_open:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a" if self.append else "w", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        """Write and flush one record so partial runs retain useful data."""
        if not self.is_open:
            self.open()
        assert self._file is not None
        payload = json.dumps(make_json_safe(record), ensure_ascii=False, sort_keys=True)
        self._file.write(payload + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the output file. Calling this method repeatedly is safe."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "JSONLWriter":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def write_jsonl(
    path: Union[str, Path], records: Iterable[Mapping[str, Any]], append: bool = False
) -> Path:
    """Write an iterable of records to JSONL and return the resolved path."""
    writer = JSONLWriter(path, append=append)
    try:
        for record in records:
            writer.write(record)
    finally:
        writer.close()
    return writer.path
