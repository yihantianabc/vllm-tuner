"""Cross-layer telemetry lifecycle aligned to a benchmark measurement window."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from .nvml_session import NVMLSample, NVMLSession, summarize_nvml_samples
from .prometheus import (
    PrometheusCollector,
    PrometheusSnapshot,
    summarize_prometheus_snapshots,
)
from .timeseries import JSONLWriter, SampleTimestamp, capture_timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelemetryFrame:
    """Engine and GPU observations sharing one alignment timestamp."""

    monotonic_ns: int
    wall_time: str
    engine: Optional[PrometheusSnapshot]
    gpu: tuple[NVMLSample, ...]
    sequence: int
    final: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSONL record with non-overlapping namespaces."""
        engine = (
            self.engine.to_dict(include_raw=self.engine.raw_text is not None)
            if self.engine is not None
            else {"available": False, "error": "Prometheus collection disabled"}
        )
        return {
            "record_type": "telemetry_sample",
            "sequence": self.sequence,
            "monotonic_ns": self.monotonic_ns,
            "wall_time": self.wall_time,
            "final": self.final,
            "client": None,
            "engine": engine,
            "gpu": {
                "available": any(sample.available for sample in self.gpu),
                "samples": [sample.to_dict() for sample in self.gpu],
            },
        }


class TelemetrySession:
    """Sample Prometheus and NVML for exactly one benchmark window.

    Start this session immediately before the measured workload and await
    :meth:`stop` before stopping vLLM. ``stop`` always requests one final sample
    and closes both collectors, even when the caller is cancelled.
    """

    def __init__(
        self,
        prometheus_endpoint: Optional[str] = None,
        device_ids: Optional[Sequence[int]] = None,
        sample_interval: float = 0.2,
        output_path: Optional[Union[str, Path]] = None,
        *,
        prometheus_url: Optional[str] = None,
        metrics_url: Optional[str] = None,
        interval_seconds: Optional[float] = None,
        jsonl_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        prometheus_collector: Optional[PrometheusCollector] = None,
        nvml_session: Optional[NVMLSession] = None,
        enable_nvml: bool = True,
        include_raw_prometheus: bool = False,
        client_metrics: Optional[Mapping[str, Any]] = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        endpoint_values = [
            value
            for value in (prometheus_endpoint, prometheus_url, metrics_url)
            if value is not None
        ]
        if len(set(endpoint_values)) > 1:
            raise ValueError("provide only one Prometheus endpoint")
        endpoint = endpoint_values[0] if endpoint_values else None
        interval = interval_seconds if interval_seconds is not None else sample_interval
        if interval <= 0:
            raise ValueError("sample_interval must be positive")

        output_values = [value for value in (output_path, jsonl_path) if value is not None]
        if len(output_values) > 1 and Path(output_values[0]) != Path(output_values[1]):
            raise ValueError("provide only one JSONL output path")
        selected_output = output_values[0] if output_values else None
        if output_dir is not None:
            directory_output = Path(output_dir) / "telemetry.jsonl"
            if selected_output is not None and Path(selected_output) != directory_output:
                raise ValueError("provide either output_dir or a JSONL output path")
            selected_output = directory_output
        elif selected_output is not None and Path(selected_output).suffix.lower() != ".jsonl":
            selected_output = Path(selected_output) / "telemetry.jsonl"

        if prometheus_collector is not None and endpoint is not None:
            raise ValueError("provide a collector or an endpoint, not both")
        self.prometheus = prometheus_collector
        if self.prometheus is None and endpoint is not None:
            self.prometheus = PrometheusCollector(endpoint, include_raw=include_raw_prometheus)

        if nvml_session is not None and not enable_nvml:
            raise ValueError("nvml_session cannot be supplied when enable_nvml is false")
        self.nvml = nvml_session
        if self.nvml is None and enable_nvml:
            self.nvml = NVMLSession(device_ids=device_ids, sample_interval=interval)

        self.sample_interval = float(interval)
        self.output_path = Path(selected_output) if selected_output is not None else None
        self._writer = JSONLWriter(self.output_path) if self.output_path is not None else None
        self._monotonic_ns = monotonic_ns
        self._wall_clock = wall_clock
        self._client_metrics = dict(client_metrics) if client_metrics is not None else None
        self._output_tokens: Optional[int] = None
        self.frames: list[TelemetryFrame] = []
        self.engine_snapshots: list[PrometheusSnapshot] = []
        self.gpu_samples: list[NVMLSample] = []
        self.errors: list[str] = []
        self.started_at: Optional[SampleTimestamp] = None
        self.ended_at: Optional[SampleTimestamp] = None
        self.final_sample_collected = False
        self._state = "created"
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self._result: Optional[dict[str, Any]] = None

    @property
    def running(self) -> bool:
        """Whether periodic collection is active."""
        return self._state == "running"

    @property
    def state(self) -> str:
        """Return the lifecycle state for diagnostics."""
        return self._state

    @property
    def result(self) -> Optional[dict[str, Any]]:
        """Return the completed namespaced result, if stopped."""
        return self._result

    def set_client_metrics(
        self, metrics: Mapping[str, Any], output_tokens: Optional[int] = None
    ) -> None:
        """Attach client results without flattening them into engine/GPU data."""
        self._client_metrics = dict(metrics)
        if output_tokens is not None:
            if output_tokens < 0:
                raise ValueError("output_tokens must be non-negative")
            self._output_tokens = output_tokens

    async def start(self) -> "TelemetrySession":
        """Open collectors, synchronously take the baseline, then start polling."""
        async with self._lifecycle_lock:
            if self._state == "running":
                raise RuntimeError("TelemetrySession is already running")
            if self._state != "created":
                raise RuntimeError("TelemetrySession instances cannot be restarted")

            self.frames.clear()
            self.engine_snapshots.clear()
            self.gpu_samples.clear()
            self.errors.clear()
            self._result = None
            self._stop_event = asyncio.Event()
            if self.prometheus is not None and hasattr(self.prometheus, "clear"):
                self.prometheus.clear()
            if self.nvml is not None:
                self.nvml.clear()
            if self._writer is not None:
                self._writer.open()

            self.started_at = self._timestamp()
            self._state = "running"
            if self.nvml is not None:
                self.nvml.open()
            try:
                await self._collect_frame(timestamp=self.started_at, final=False)
            except BaseException:
                # Start is transactional: never leave NVML or an artifact file
                # open if an unexpected fatal exception escapes collection.
                await self._close_resources()
                self._state = "stopped"
                raise
            self._task = asyncio.create_task(self._sampling_loop(), name="cross-layer-telemetry")
            return self

    def _timestamp(self) -> SampleTimestamp:
        return capture_timestamp(self._monotonic_ns, self._wall_clock)

    async def _collect_engine(self, timestamp: SampleTimestamp) -> Optional[PrometheusSnapshot]:
        if self.prometheus is None:
            return None
        try:
            collect = self.prometheus.collect(timestamp=timestamp)
            snapshot = await collect if inspect.isawaitable(collect) else collect
            if not isinstance(snapshot, PrometheusSnapshot):
                raise TypeError("Prometheus collector returned an invalid snapshot")
            self.engine_snapshots.append(snapshot)
            return snapshot
        except asyncio.CancelledError:
            raise
        except Exception as error:
            message = f"Prometheus sample failed: {type(error).__name__}: {error}"
            self.errors.append(message)
            snapshot = PrometheusSnapshot(
                monotonic_ns=timestamp.monotonic_ns,
                wall_time=timestamp.wall_time,
                error=message,
            )
            self.engine_snapshots.append(snapshot)
            return snapshot

    def _collect_gpu(self, timestamp: SampleTimestamp) -> tuple[NVMLSample, ...]:
        if self.nvml is None:
            return ()
        try:
            samples = tuple(self.nvml.sample_once(timestamp=timestamp))
            self.gpu_samples.extend(samples)
            return samples
        except Exception as error:
            message = f"NVML sample failed: {type(error).__name__}: {error}"
            self.errors.append(message)
            return ()

    async def _collect_frame(
        self, timestamp: Optional[SampleTimestamp] = None, final: bool = False
    ) -> TelemetryFrame:
        captured = timestamp or self._timestamp()
        # Read the local driver first so a slow /metrics response does not move
        # GPU sampling outside the intended window boundary.
        gpu_samples = self._collect_gpu(captured)
        engine_snapshot = await self._collect_engine(captured)
        frame = TelemetryFrame(
            monotonic_ns=captured.monotonic_ns,
            wall_time=captured.wall_time,
            engine=engine_snapshot,
            gpu=gpu_samples,
            sequence=len(self.frames),
            final=final,
        )
        self.frames.append(frame)
        if final:
            self.final_sample_collected = True
        if self._writer is not None:
            try:
                self._writer.write(frame.to_dict())
            except Exception as error:
                message = f"JSONL write failed: {type(error).__name__}: {error}"
                self.errors.append(message)
                logger.error(message, exc_info=True)
                self._writer.close()
                self._writer = None
        return frame

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
                    await self._collect_frame()
                    deadline += self.sample_interval
                    if deadline <= loop.time():
                        deadline = loop.time() + self.sample_interval
        except asyncio.CancelledError:
            raise
        except Exception as error:
            message = f"telemetry loop failed: {type(error).__name__}: {error}"
            self.errors.append(message)
            logger.warning(message, exc_info=True)

    async def stop(
        self,
        client_metrics: Optional[Mapping[str, Any]] = None,
        output_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Reliably stop and return ``client``/``engine``/``gpu`` namespaces.

        Cleanup runs in a shielded task. If the caller is cancelled, collection
        still takes the final sample and closes resources before cancellation is
        re-raised to the caller.
        """
        if client_metrics is not None:
            self._client_metrics = dict(client_metrics)
        if output_tokens is not None:
            if output_tokens < 0:
                raise ValueError("output_tokens must be non-negative")
            self._output_tokens = output_tokens
        cleanup_task = asyncio.create_task(self._stop_impl())
        try:
            return await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
            raise

    async def _stop_impl(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self._state == "created":
                raise RuntimeError("TelemetrySession has not been started")
            if self._state == "stopped":
                assert self._result is not None
                return self._result

            self._state = "stopping"
            self._stop_event.set()
            task = self._task
            self._task = None
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            try:
                await self._collect_frame(final=True)
            except asyncio.CancelledError:
                # _stop_impl itself is shielded, but retain a defensive cleanup
                # path for direct task cancellation by an integration.
                self.errors.append("final telemetry sample was cancelled")
            except Exception as error:
                self.errors.append(
                    f"final telemetry sample failed: {type(error).__name__}: {error}"
                )
            finally:
                self.ended_at = self._timestamp()
                await self._close_resources()

            self._result = self._build_result()
            self._state = "stopped"
            return self._result

    async def _close_resources(self) -> None:
        if self.nvml is not None:
            try:
                self.nvml.close()
            except Exception as error:
                self.errors.append(f"NVML close failed: {type(error).__name__}: {error}")
        if self.prometheus is not None and hasattr(self.prometheus, "close"):
            try:
                close_result = self.prometheus.close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as error:
                self.errors.append(f"Prometheus close failed: {type(error).__name__}: {error}")
        if self._writer is not None:
            self._writer.close()

    def _resolved_output_tokens(self) -> Optional[int]:
        if self._output_tokens is not None:
            return self._output_tokens
        if self._client_metrics is None:
            return None
        candidates: list[Mapping[str, Any]] = [self._client_metrics]
        aggregate = self._client_metrics.get("aggregate")
        if isinstance(aggregate, Mapping):
            candidates.append(aggregate)
        for metrics in candidates:
            for key in (
                "output_tokens",
                "total_output_tokens",
                "completed_output_tokens",
            ):
                value = metrics.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
        return None

    def _build_result(self) -> dict[str, Any]:
        client = dict(self._client_metrics) if self._client_metrics is not None else {}
        client.setdefault("available", self._client_metrics is not None)
        engine = summarize_prometheus_snapshots(self.engine_snapshots)
        gpu = summarize_nvml_samples(
            self.gpu_samples,
            output_tokens=self._resolved_output_tokens(),
            session_errors=self.nvml.errors if self.nvml is not None else None,
        )
        duration_seconds: Optional[float] = None
        if self.started_at is not None and self.ended_at is not None:
            duration_seconds = max(
                0.0,
                (self.ended_at.monotonic_ns - self.started_at.monotonic_ns) / 1_000_000_000.0,
            )
        return {
            "client": client,
            "engine": engine,
            "gpu": gpu,
            "window": {
                "available": self.started_at is not None and self.ended_at is not None,
                "start_monotonic_ns": (
                    self.started_at.monotonic_ns if self.started_at is not None else None
                ),
                "end_monotonic_ns": (
                    self.ended_at.monotonic_ns if self.ended_at is not None else None
                ),
                "start_wall_time": (self.started_at.wall_time if self.started_at else None),
                "end_wall_time": self.ended_at.wall_time if self.ended_at else None,
                "duration_seconds": duration_seconds,
                "sample_interval_seconds": self.sample_interval,
                "frame_count": len(self.frames),
                "final_sample_collected": self.final_sample_collected,
                "errors": list(self.errors),
                "jsonl_path": (str(self.output_path) if self.output_path is not None else None),
            },
        }

    def get_summary(self) -> dict[str, Any]:
        """Return a live or completed summary without flattening namespaces."""
        return self._result if self._result is not None else self._build_result()

    async def __aenter__(self) -> "TelemetrySession":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.stop()
