"""Tests for the cross-layer measurement-window lifecycle."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from vllm_tuner.profiling.nvml_session import NVMLSession
from vllm_tuner.profiling.prometheus import PrometheusCollector
from vllm_tuner.profiling.session import TelemetrySession


class SessionNVML:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_count = 0

    def nvmlInit(self) -> None:
        self.initialized = True

    def nvmlShutdown(self) -> None:
        self.initialized = False
        self.shutdown_count += 1

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(used=256 * 1024**2, total=1024 * 1024**2)

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(gpu=75, memory=25)

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        return 65

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return 150_000

    def nvmlDeviceGetClockInfo(self, handle: int, clock: int) -> int:
        return 2500


def _metrics(counter: int) -> str:
    return f"""
vllm:num_requests_running 2
vllm:num_requests_waiting 1
vllm:kv_cache_usage_perc 0.5
vllm:prompt_tokens_total {counter}
vllm:generation_tokens_total {counter // 2}
vllm:num_preemptions_total 4
"""


@pytest.mark.asyncio
async def test_session_aligns_namespaces_persists_jsonl_and_uses_final_delta(
    tmp_path,
) -> None:
    responses = iter([_metrics(100), _metrics(140)])

    async def fetch() -> str:
        return next(responses)

    nvml_module = SessionNVML()
    session = TelemetrySession(
        prometheus_collector=PrometheusCollector("http://vllm", fetcher=fetch),
        nvml_session=NVMLSession(device_ids=[0], nvml_module=nvml_module),
        sample_interval=60,
        output_path=tmp_path / "telemetry.jsonl",
    )

    await session.start()
    result = await session.stop(client_metrics={"completed": 2}, output_tokens=20)

    assert set(result) == {"client", "engine", "gpu", "window"}
    assert result["client"]["completed"] == 2
    assert result["engine"]["prompt_tokens_total"]["delta"] == 40.0
    assert result["gpu"]["peak_memory_mb"] == 256.0
    assert result["window"]["final_sample_collected"] is True
    assert nvml_module.shutdown_count == 1

    records = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[-1]["final"] is True
    assert set(records[-1]) >= {"client", "engine", "gpu", "monotonic_ns", "wall_time"}
    assert records[-1]["engine"]["metrics"]["prompt_tokens_total"] == 140.0


@pytest.mark.asyncio
async def test_context_manager_stops_after_benchmark_exception() -> None:
    counter = 0

    async def fetch() -> str:
        nonlocal counter
        counter += 1
        return _metrics(counter)

    nvml_module = SessionNVML()
    session = TelemetrySession(
        prometheus_collector=PrometheusCollector("http://vllm", fetcher=fetch),
        nvml_session=NVMLSession(device_ids=[0], nvml_module=nvml_module),
        sample_interval=60,
    )

    with pytest.raises(RuntimeError, match="benchmark failed"):
        async with session:
            raise RuntimeError("benchmark failed")

    assert session.state == "stopped"
    assert session.final_sample_collected is True
    assert nvml_module.shutdown_count == 1


@pytest.mark.asyncio
async def test_context_manager_cancellation_reliably_exits_and_collects_final_sample() -> None:
    async def fetch() -> str:
        return _metrics(10)

    nvml_module = SessionNVML()
    session = TelemetrySession(
        prometheus_collector=PrometheusCollector("http://vllm", fetcher=fetch),
        nvml_session=NVMLSession(device_ids=[0], nvml_module=nvml_module),
        sample_interval=60,
    )
    entered = asyncio.Event()

    async def measured_work() -> None:
        async with session:
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(measured_work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.state == "stopped"
    assert session.frames[-1].final is True
    assert nvml_module.shutdown_count == 1


@pytest.mark.asyncio
async def test_disabled_sources_are_explicitly_unavailable() -> None:
    session = TelemetrySession(enable_nvml=False, sample_interval=60)

    await session.start()
    result = await session.stop()

    assert result["engine"]["available"] is False
    assert result["gpu"]["available"] is False
    assert result["gpu"]["peak_memory_mb"] is None
    assert result["client"] == {"available": False}
    assert session.sample_interval == 60
