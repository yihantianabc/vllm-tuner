"""Tests for continuous NVML collection without requiring a physical GPU."""

from types import SimpleNamespace

import pytest

from vllm_tuner.profiling.nvml_session import (
    NVMLSample,
    NVMLSession,
    summarize_nvml_samples,
)
from vllm_tuner.profiling.timeseries import SampleTimestamp


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_count = 0
        self.tick = 0

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
        used_mb = [100.0, 200.0, 300.0][min(self.tick, 2)]
        return SimpleNamespace(used=used_mb * 1024**2, total=1000 * 1024**2)

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(gpu=20 + self.tick * 10, memory=10 + self.tick)

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        return 60 + self.tick

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return (100 + self.tick * 50) * 1000

    def nvmlDeviceGetClockInfo(self, handle: int, clock: int) -> int:
        return 2000 if clock == self.NVML_CLOCK_SM else 3000


def test_nvml_sample_contains_all_required_sensors_and_timestamps() -> None:
    fake = FakeNVML()
    session = NVMLSession(device_ids=[0], nvml_module=fake)
    timestamp = SampleTimestamp(123, "2026-08-15T00:00:00+00:00")

    sample = session.sample_once(timestamp)[0]
    session.close()

    assert sample.monotonic_ns == 123
    assert sample.wall_time == timestamp.wall_time
    assert sample.memory_used_mb == pytest.approx(100.0)
    assert sample.memory_total_mb == pytest.approx(1000.0)
    assert sample.gpu_utilization_percent == 20.0
    assert sample.memory_utilization_percent == 10.0
    assert sample.power_w == 100.0
    assert sample.temperature_c == 60.0
    assert sample.sm_clock_mhz == 2000.0
    assert sample.memory_clock_mhz == 3000.0
    assert fake.shutdown_count == 1


def test_nvml_summary_computes_peak_mean_p95_and_energy_per_token() -> None:
    samples = [
        NVMLSample(0, 0, "a", memory_used_mb=100, gpu_utilization_percent=10, power_w=100),
        NVMLSample(
            0,
            1_000_000_000,
            "b",
            memory_used_mb=200,
            gpu_utilization_percent=50,
            power_w=200,
        ),
        NVMLSample(
            0,
            2_000_000_000,
            "c",
            memory_used_mb=300,
            gpu_utilization_percent=90,
            power_w=200,
        ),
    ]

    summary = summarize_nvml_samples(samples, output_tokens=70)

    assert summary["peak_memory_mb"] == 300.0
    assert summary["mean_memory_mb"] == 200.0
    assert summary["p95_memory_mb"] == pytest.approx(290.0)
    assert summary["mean_gpu_utilization_percent"] == 50.0
    assert summary["energy_joules"] == pytest.approx(350.0)
    assert summary["energy_per_output_token_joules"] == pytest.approx(5.0)


def test_nvml_initialization_failure_is_unavailable_not_zero() -> None:
    class BrokenNVML(FakeNVML):
        def nvmlInit(self) -> None:
            raise RuntimeError("driver missing")

    session = NVMLSession(device_ids=[0], nvml_module=BrokenNVML())
    samples = session.sample_once(SampleTimestamp(1, "wall"))
    summary = session.summary()

    assert samples[0].available is False
    assert "driver missing" in samples[0].error
    assert summary["available"] is False
    assert summary["peak_memory_mb"] is None


@pytest.mark.asyncio
async def test_async_nvml_session_takes_final_sample_before_shutdown() -> None:
    fake = FakeNVML()
    session = NVMLSession(device_ids=[0], sample_interval=60, nvml_module=fake)

    await session.start()
    summary = await session.stop()

    assert summary["sample_count"] == 2
    assert fake.shutdown_count == 1
    assert session.running is False

    repeated_summary = await session.stop()
    assert repeated_summary["sample_count"] == 2
    assert fake.shutdown_count == 1
