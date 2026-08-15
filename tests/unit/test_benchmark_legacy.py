"""Regression tests for legacy benchmark compatibility facades."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from vllm_tuner.baseline.runner import VLLMBaselineRunner
from vllm_tuner.benchmarks.metrics import aggregate_request_results
from vllm_tuner.benchmarks.models import (
    BenchmarkResult,
    RequestResult,
    RequestSpec,
    RequestStatus,
)
from vllm_tuner.benchmarks.request_generator import (
    BenchmarkClient,
    BenchmarkRequest,
    BenchmarkRunner,
    RequestGenerator,
)
from vllm_tuner.benchmarks.sse_client import SSEBenchmarkClient
from vllm_tuner.config.models import GPUConfig, TuningConfig, WorkloadConfig
from vllm_tuner.profiling.vllm_metrics import VLLMMetrics, VLLMMetricsTracker


def _request_result(request_id: str, e2e_ms: int, *, sent_at: int = 1_000_000_000) -> RequestResult:
    first_token_at = sent_at + 10_000_000
    return RequestResult(
        request_id=request_id,
        scheduled_at=sent_at,
        sent_at=sent_at,
        first_token_at=first_token_at,
        finished_at=sent_at + e2e_ms * 1_000_000,
        input_tokens=3,
        output_tokens=3,
        token_timestamps=[first_token_at, first_token_at + 20_000_000],
        status=RequestStatus.SUCCESS,
        token_count_source="usage",
    )


def _benchmark_result(count: int = 2) -> BenchmarkResult:
    requests = [_request_result(f"request-{index}", 100 + index * 100) for index in range(count)]
    started_at = 1_000_000_000
    finished_at = 2_000_000_000
    return BenchmarkResult(
        backend="sse",
        started_at=started_at,
        finished_at=finished_at,
        request_results=requests,
        aggregate=aggregate_request_results(
            requests, started_at=started_at, finished_at=finished_at
        ),
    )


def _config() -> TuningConfig:
    return TuningConfig(
        model="gpt2",
        gpu=GPUConfig(device_ids=[0]),
        workload=WorkloadConfig(
            sample_size=2,
            max_tokens=3,
            max_concurrency=2,
            warmup_requests=1,
        ),
    )


def test_request_generator_exposes_typed_conversion() -> None:
    generator = RequestGenerator(["a", "b"], max_tokens=17)

    legacy = generator.generate_requests()
    specs = generator.generate_specs("model")

    assert [request.request_id for request in legacy] == ["req_0", "req_1"]
    assert [spec.request_id for spec in specs] == ["req_0", "req_1"]
    assert specs[0].model == "model"
    assert specs[0].max_tokens == 17


def test_legacy_metrics_use_real_e2e_and_numpy_percentile() -> None:
    tracker = VLLMMetricsTracker()
    requests = [
        _request_result(f"request-{index}", e2e_ms)
        for index, e2e_ms in enumerate((100, 200, 300, 400))
    ]
    result = BenchmarkResult(
        backend="sse",
        started_at=1_000_000_000,
        finished_at=2_000_000_000,
        request_results=requests,
    )

    tracker.record_benchmark_result(result)
    summary = tracker.get_summary()

    assert summary["avg_latency_ms"] == 250.0
    assert summary["p50_latency_ms"] == 250.0
    assert summary["p95_latency_ms"] == pytest.approx(385.0)
    assert summary["p99_latency_ms"] == pytest.approx(397.0)
    assert summary["avg_ttft_ms"] == 10.0
    assert len(summary["request_results"]) == 4


def test_legacy_metrics_never_approximate_e2e_from_ttft_and_tpot() -> None:
    metrics = VLLMMetrics()
    metrics.ttft_times = [0.1]
    metrics.tpot_times = [0.02]

    summary = metrics.to_dict()

    assert summary["avg_ttft_ms"] == 100.0
    assert summary["avg_tpot_ms"] == 20.0
    assert summary["avg_latency_ms"] == 0.0


@pytest.mark.asyncio
async def test_benchmark_runner_delegates_warmup_and_concurrency(monkeypatch) -> None:
    captured = {}
    core_result = _benchmark_result()

    async def fake_run(
        self: SSEBenchmarkClient, requests: list[RequestSpec], **kwargs: Any
    ) -> BenchmarkResult:
        captured["requests"] = requests
        captured.update(kwargs)
        return core_result

    monkeypatch.setattr(SSEBenchmarkClient, "run", fake_run)
    launcher = SimpleNamespace(base_url="http://vllm", config=SimpleNamespace(model="model"))
    runner = BenchmarkRunner(launcher, concurrency=2, warmup_requests=3, max_tokens=11)

    summary = await runner.run_benchmark(["a", "b"])

    assert captured["warmup_requests"] == 3
    assert captured["max_concurrency"] == 2
    assert captured["requests"][0].max_tokens == 11
    assert summary["requests_completed"] == 2
    assert summary["total_input_tokens"] == 6
    assert runner.last_result is core_result


@pytest.mark.asyncio
async def test_benchmark_client_records_typed_result(monkeypatch) -> None:
    typed_result = _request_result("legacy", 120)

    async def fake_send(
        self: SSEBenchmarkClient,
        request: RequestSpec,
        http_client: httpx.AsyncClient,
        **kwargs: Any,
    ) -> RequestResult:
        return typed_result

    monkeypatch.setattr(SSEBenchmarkClient, "send_request", fake_send)
    tracker = VLLMMetricsTracker()
    client = BenchmarkClient("http://vllm", tracker, model_name="model")

    async with httpx.AsyncClient() as http_client:
        result = await client.send_request(BenchmarkRequest("legacy", "prompt"), http_client)

    assert result is not None
    assert result["duration"] == 0.12
    assert tracker.get_summary()["requests_completed"] == 1


@pytest.mark.asyncio
async def test_baseline_runner_uses_streaming_core(monkeypatch, tmp_path) -> None:
    captured = {}
    core_result = _benchmark_result()

    async def fake_run(
        self: SSEBenchmarkClient, requests: list[RequestSpec], **kwargs: Any
    ) -> BenchmarkResult:
        captured["requests"] = requests
        captured.update(kwargs)
        return core_result

    monkeypatch.setattr(SSEBenchmarkClient, "run", fake_run)
    runner = VLLMBaselineRunner(_config(), tmp_path)
    tracker = VLLMMetricsTracker()

    summary = await runner._run_benchmark(["a", "b"], tracker)

    assert captured["max_concurrency"] == 2
    assert all(spec.max_tokens == 3 for spec in captured["requests"])
    assert summary["requests_completed"] == 2
    assert runner.last_benchmark_result is core_result


@pytest.mark.asyncio
async def test_baseline_measurement_uses_namespaced_telemetry(monkeypatch, tmp_path) -> None:
    instances = []

    class FakeTelemetrySession:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.running = False
            self.gpu_samples = [
                SimpleNamespace(
                    memory_used_mb=512.0,
                    gpu_utilization_percent=75.0,
                )
            ]
            instances.append(self)

        async def start(self) -> "FakeTelemetrySession":
            self.running = True
            return self

        async def stop(self, **kwargs: Any) -> dict[str, Any]:
            self.running = False
            return {
                "client": kwargs.get("client_metrics"),
                "engine": {"available": True},
                "gpu": {"available": True, "peak_memory_mb": 512.0},
                "window": {"final_sample_collected": True},
            }

    monkeypatch.setattr("vllm_tuner.baseline.runner.TelemetrySession", FakeTelemetrySession)
    runner = VLLMBaselineRunner(_config(), tmp_path)
    runner._start_vllm_server = AsyncMock()
    runner._stop_server = AsyncMock()
    runner._load_prompts = Mock(return_value=["warmup", "a", "b"])
    benchmark_summary = {
        "requests_completed": 2,
        "num_requests": 2,
        "total_output_tokens": 6,
    }
    runner._run_benchmark = AsyncMock(return_value=benchmark_summary)
    runner.gpu_collector.collect_all = Mock(return_value=[])
    runner._generate_outputs = Mock()

    await runner.run()

    assert len(instances) == 1
    assert instances[0].kwargs["prometheus_endpoint"].endswith("/metrics")
    assert runner.telemetry_result["window"]["final_sample_collected"] is True
    assert runner.metrics.metrics["engine"] == {"available": True}
    assert runner.metrics.memory_samples == [512.0]
    assert runner.metrics.gpu_util_samples == [0.75]
