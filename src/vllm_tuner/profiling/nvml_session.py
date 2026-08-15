"""Continuous, window-scoped NVIDIA GPU telemetry using NVML."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

import pynvml  # type: ignore[import-untyped]

from .timeseries import (
    SampleTimestamp,
    capture_timestamp,
    integrate_power_samples,
    summarize_values,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NVMLSample:
    """One GPU sample with explicit optional values for unsupported sensors."""

    device_id: int
    monotonic_ns: int
    wall_time: str
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    power_w: Optional[float] = None
    temperature_c: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        """Whether at least one requested sensor was read successfully."""
        return any(
            value is not None
            for value in (
                self.memory_used_mb,
                self.memory_total_mb,
                self.gpu_utilization_percent,
                self.memory_utilization_percent,
                self.power_w,
                self.temperature_c,
                self.sm_clock_mhz,
                self.memory_clock_mhz,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "device_id": self.device_id,
            "monotonic_ns": self.monotonic_ns,
            "wall_time": self.wall_time,
            "available": self.available,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "power_w": self.power_w,
            "temperature_c": self.temperature_c,
            "sm_clock_mhz": self.sm_clock_mhz,
            "memory_clock_mhz": self.memory_clock_mhz,
            "error": self.error,
        }


class NVMLSession:
    """Own NVML lifecycle and optionally run a periodic sampling task.

    ``open``/``sample_once``/``close`` are exposed for a parent
    :class:`TelemetrySession` that aligns GPU and engine samples. ``start`` and
    ``stop`` provide a standalone asynchronous session for direct use.
    """

    def __init__(
        self,
        device_ids: Optional[Sequence[int]] = None,
        sample_interval: float = 0.2,
        nvml_module: Any = pynvml,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        raise_on_error: bool = False,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.requested_device_ids = None if device_ids is None else list(device_ids)
        if self.requested_device_ids is not None and any(
            device_id < 0 for device_id in self.requested_device_ids
        ):
            raise ValueError("device IDs must be non-negative")
        self.device_ids: list[int] = list(self.requested_device_ids or [])
        self.sample_interval = sample_interval
        self._nvml = nvml_module
        self._monotonic_ns = monotonic_ns
        self._wall_clock = wall_clock
        self.raise_on_error = raise_on_error
        self.initialized = False
        self.initialization_error: Optional[str] = None
        self.samples: list[NVMLSample] = []
        self.errors: list[str] = []
        self._handles: dict[int, Any] = {}
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._standalone_state = "created"
        self._lifecycle_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        """Whether the standalone periodic sampler is active."""
        return self._task is not None and not self._task.done()

    def clear(self) -> None:
        """Clear samples and recoverable errors from an earlier window."""
        self.samples.clear()
        self.errors.clear()

    def open(self) -> bool:
        """Initialise NVML and resolve device handles.

        By default an unavailable driver degrades telemetry instead of failing a
        benchmark. Set ``raise_on_error`` when GPU telemetry is mandatory.
        """
        if self.initialized:
            return True
        self.initialization_error = None
        try:
            self._nvml.nvmlInit()
            self.initialized = True
            device_count = int(self._nvml.nvmlDeviceGetCount())
            if self.requested_device_ids is None:
                self.device_ids = list(range(device_count))
            else:
                self.device_ids = list(self.requested_device_ids)
            for device_id in self.device_ids:
                if device_id >= device_count:
                    raise ValueError(
                        f"GPU device {device_id} is unavailable; NVML reports {device_count} device(s)"
                    )
                self._handles[device_id] = self._nvml.nvmlDeviceGetHandleByIndex(device_id)
            return True
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.initialization_error = message
            self.errors.append(message)
            logger.warning("NVML telemetry unavailable: %s", message)
            if self.initialized:
                with contextlib.suppress(Exception):
                    self._nvml.nvmlShutdown()
                self.initialized = False
            self._handles.clear()
            if self.raise_on_error:
                raise
            return False

    def close(self) -> None:
        """Shutdown NVML. Calling this method repeatedly is safe."""
        if not self.initialized:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.errors.append(message)
            if self.raise_on_error:
                raise
            logger.warning("NVML shutdown failed: %s", message)
        finally:
            self.initialized = False
            self._handles.clear()

    def _optional_query(self, function_name: str, *args: Any) -> Optional[Any]:
        try:
            function = getattr(self._nvml, function_name)
            return function(*args)
        except Exception:
            return None

    def _sample_device(self, device_id: int, timestamp: SampleTimestamp) -> NVMLSample:
        handle = self._handles.get(device_id)
        if handle is None:
            return NVMLSample(
                device_id=device_id,
                monotonic_ns=timestamp.monotonic_ns,
                wall_time=timestamp.wall_time,
                error=self.initialization_error or "NVML device handle unavailable",
            )

        memory_info = self._optional_query("nvmlDeviceGetMemoryInfo", handle)
        utilization = self._optional_query("nvmlDeviceGetUtilizationRates", handle)
        temperature = self._optional_query(
            "nvmlDeviceGetTemperature",
            handle,
            getattr(self._nvml, "NVML_TEMPERATURE_GPU", 0),
        )
        power_usage = self._optional_query("nvmlDeviceGetPowerUsage", handle)
        sm_clock = self._optional_query(
            "nvmlDeviceGetClockInfo", handle, getattr(self._nvml, "NVML_CLOCK_SM", 1)
        )
        memory_clock = self._optional_query(
            "nvmlDeviceGetClockInfo", handle, getattr(self._nvml, "NVML_CLOCK_MEM", 2)
        )

        memory_used_mb: Optional[float] = None
        memory_total_mb: Optional[float] = None
        if memory_info is not None:
            try:
                memory_used_mb = float(memory_info.used) / (1024.0 * 1024.0)
                memory_total_mb = float(memory_info.total) / (1024.0 * 1024.0)
            except (AttributeError, TypeError, ValueError):
                memory_used_mb = None
                memory_total_mb = None

        gpu_utilization: Optional[float] = None
        memory_utilization: Optional[float] = None
        if utilization is not None:
            try:
                gpu_utilization = float(utilization.gpu)
                memory_utilization = float(utilization.memory)
            except (AttributeError, TypeError, ValueError):
                gpu_utilization = None
                memory_utilization = None

        sample = NVMLSample(
            device_id=device_id,
            monotonic_ns=timestamp.monotonic_ns,
            wall_time=timestamp.wall_time,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            gpu_utilization_percent=gpu_utilization,
            memory_utilization_percent=memory_utilization,
            power_w=_scaled_float(power_usage, 1000.0),
            temperature_c=_scaled_float(temperature, 1.0),
            sm_clock_mhz=_scaled_float(sm_clock, 1.0),
            memory_clock_mhz=_scaled_float(memory_clock, 1.0),
        )
        if sample.available:
            return sample
        return replace(sample, error="all NVML sensor queries failed")

    def sample_once(self, timestamp: Optional[SampleTimestamp] = None) -> list[NVMLSample]:
        """Collect all requested GPUs at one shared timestamp."""
        captured = timestamp or capture_timestamp(self._monotonic_ns, self._wall_clock)
        if not self.initialized:
            self.open()

        if not self.initialized:
            target_ids = self.device_ids
            if not target_ids and self.requested_device_ids is not None:
                target_ids = list(self.requested_device_ids)
            unavailable = [
                NVMLSample(
                    device_id=device_id,
                    monotonic_ns=captured.monotonic_ns,
                    wall_time=captured.wall_time,
                    error=self.initialization_error or "NVML unavailable",
                )
                for device_id in target_ids
            ]
            self.samples.extend(unavailable)
            return unavailable

        collected = [self._sample_device(device_id, captured) for device_id in self.device_ids]
        self.samples.extend(collected)
        return collected

    # ``collect`` is intentionally synchronous: NVML calls are short local
    # driver queries and a parent sampler needs exact timestamp alignment.
    collect = sample_once

    async def start(self) -> "NVMLSession":
        """Start standalone periodic sampling with an immediate baseline."""
        async with self._lifecycle_lock:
            if self._standalone_state == "running":
                raise RuntimeError("NVMLSession is already running")
            if self._standalone_state != "created":
                raise RuntimeError("NVMLSession instances cannot be restarted")
            self.clear()
            self._stop_event = asyncio.Event()
            self.open()
            self.sample_once()
            self._task = asyncio.create_task(self._sampling_loop(), name="nvml-telemetry")
            self._standalone_state = "running"
            return self

    async def _sampling_loop(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.sample_interval
        try:
            while not self._stop_event.is_set():
                timeout = max(0.0, deadline - loop.time())
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
                    break
                except asyncio.TimeoutError:
                    self.sample_once()
                    deadline += self.sample_interval
                    if deadline <= loop.time():
                        deadline = loop.time() + self.sample_interval
        except asyncio.CancelledError:
            raise

    async def stop(self, final_sample: bool = True) -> dict[str, Any]:
        """Stop sampling, take a final sample, shutdown NVML, and aggregate."""
        cleanup_task = asyncio.create_task(self._stop_impl(final_sample))
        try:
            return await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
            raise

    async def _stop_impl(self, final_sample: bool) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self._standalone_state == "created":
                raise RuntimeError("NVMLSession has not been started")
            if self._standalone_state == "stopped":
                return self.summary()
            self._standalone_state = "stopping"
            self._stop_event.set()
            task = self._task
            self._task = None
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            try:
                if final_sample:
                    self.sample_once()
            finally:
                self.close()
                self._standalone_state = "stopped"
            return self.summary()

    def summary(self, output_tokens: Optional[int] = None) -> dict[str, Any]:
        """Aggregate samples, optionally deriving energy per output token."""
        return summarize_nvml_samples(
            self.samples,
            output_tokens=output_tokens,
            session_errors=self.errors,
        )

    async def __aenter__(self) -> "NVMLSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.stop(final_sample=True)


def _scaled_float(value: Any, divisor: float) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value) / divisor
    except (TypeError, ValueError):
        return None
    return numeric if math_is_finite(numeric) else None


def math_is_finite(value: float) -> bool:
    """Local helper kept mock-friendly and explicit."""
    return value != float("inf") and value != float("-inf") and value == value


_NVML_FIELDS = (
    "memory_used_mb",
    "memory_total_mb",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "power_w",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
)


def _summarize_device(
    samples: Sequence[NVMLSample], output_tokens: Optional[int]
) -> dict[str, Any]:
    metrics = {
        field_name: summarize_values(getattr(sample, field_name) for sample in samples)
        for field_name in _NVML_FIELDS
    }
    power_samples = [(sample.monotonic_ns, sample.power_w) for sample in samples]
    energy = integrate_power_samples(power_samples)
    energy_per_token = (
        energy / output_tokens
        if energy is not None and output_tokens is not None and output_tokens > 0
        else None
    )
    return {
        "available": any(sample.available for sample in samples),
        "sample_count": len(samples),
        "errors": [sample.error for sample in samples if sample.error],
        **metrics,
        "energy_joules": energy,
        "energy_per_output_token_joules": energy_per_token,
    }


def summarize_nvml_samples(
    samples: Iterable[NVMLSample],
    output_tokens: Optional[int] = None,
    session_errors: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Aggregate per-device and total GPU time series."""
    sample_list = sorted(samples, key=lambda sample: (sample.monotonic_ns, sample.device_id))
    by_device: dict[int, list[NVMLSample]] = {}
    by_timestamp: dict[int, list[NVMLSample]] = {}
    for sample in sample_list:
        by_device.setdefault(sample.device_id, []).append(sample)
        by_timestamp.setdefault(sample.monotonic_ns, []).append(sample)

    devices = {
        str(device_id): _summarize_device(device_samples, output_tokens)
        for device_id, device_samples in sorted(by_device.items())
    }

    total_fields: dict[str, list[Optional[float]]] = {field_name: [] for field_name in _NVML_FIELDS}
    for timestamp_samples in by_timestamp.values():
        for field_name in ("memory_used_mb", "memory_total_mb", "power_w"):
            values = [
                getattr(sample, field_name)
                for sample in timestamp_samples
                if getattr(sample, field_name) is not None
            ]
            total_fields[field_name].append(sum(values) if values else None)
        for field_name in (
            "gpu_utilization_percent",
            "memory_utilization_percent",
            "sm_clock_mhz",
            "memory_clock_mhz",
        ):
            values = [
                getattr(sample, field_name)
                for sample in timestamp_samples
                if getattr(sample, field_name) is not None
            ]
            total_fields[field_name].append(sum(values) / len(values) if values else None)
        temperatures = [
            sample.temperature_c for sample in timestamp_samples if sample.temperature_c is not None
        ]
        total_fields["temperature_c"].append(max(temperatures) if temperatures else None)

    aggregate = {
        field_name: summarize_values(values) for field_name, values in total_fields.items()
    }
    device_energies = [
        device_summary["energy_joules"]
        for device_summary in devices.values()
        if device_summary["energy_joules"] is not None
    ]
    energy = sum(device_energies) if device_energies else None
    energy_per_token = (
        energy / output_tokens
        if energy is not None and output_tokens is not None and output_tokens > 0
        else None
    )
    memory_summary = aggregate["memory_used_mb"]
    utilization_summary = aggregate["gpu_utilization_percent"]
    errors = list(session_errors or []) + [sample.error for sample in sample_list if sample.error]
    return {
        "available": any(sample.available for sample in sample_list),
        "sample_count": len(sample_list),
        "device_count": len(by_device),
        "errors": errors,
        "devices": devices,
        "aggregate": aggregate,
        **aggregate,
        "peak_memory_mb": memory_summary["peak"],
        "mean_memory_mb": memory_summary["mean"],
        "p95_memory_mb": memory_summary["p95"],
        "peak_gpu_utilization_percent": utilization_summary["peak"],
        "mean_gpu_utilization_percent": utilization_summary["mean"],
        "p95_gpu_utilization_percent": utilization_summary["p95"],
        "energy_joules": energy,
        "energy_per_output_token_joules": energy_per_token,
    }


# A concise alias used in a few integrations and test fixtures.
GPUSample = NVMLSample
