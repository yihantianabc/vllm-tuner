"""Compatibility request API backed by the trustworthy streaming client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

from vllm_tuner.profiling.vllm_metrics import VLLMMetricsTracker
from vllm_tuner.vllm.launcher import VLLMLauncher

from .models import BenchmarkResult, RequestSpec
from .sse_client import SSEBenchmarkClient, TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRequest:
    """Historical request shape retained for source compatibility."""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0

    def to_request_spec(self, model: Optional[str] = None) -> RequestSpec:
        """Convert to the typed benchmark-core request."""

        return RequestSpec(
            request_id=self.request_id,
            prompt=self.prompt,
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )


class RequestGenerator:
    """Generate legacy or typed requests from prompts."""

    def __init__(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate_requests(self) -> list[BenchmarkRequest]:
        """Generate requests using the historical public type."""

        return [
            BenchmarkRequest(
                request_id=f"req_{index}",
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            for index, prompt in enumerate(self.prompts)
        ]

    def generate_specs(self, model: Optional[str] = None) -> list[RequestSpec]:
        """Generate typed requests for the new streaming core."""

        return [request.to_request_spec(model) for request in self.generate_requests()]

    async def async_requests(self) -> AsyncIterator[BenchmarkRequest]:
        """Iterate over historical request objects asynchronously."""

        for request in self.generate_requests():
            yield request


class BenchmarkClient:
    """Historical client facade delegating to :class:`SSEBenchmarkClient`."""

    def __init__(
        self,
        base_url: str,
        metrics_tracker: VLLMMetricsTracker,
        model_name: str = "gpt2",
        timeout: int = 300,
        *,
        token_counter: Optional[TokenCounter] = None,
        tokenizer: Optional[object] = None,
    ) -> None:
        self.base_url = base_url
        self.metrics_tracker = metrics_tracker
        self.model_name = model_name
        self.timeout = timeout
        self.streaming_client = SSEBenchmarkClient(
            base_url,
            model_name,
            timeout=float(timeout),
            token_counter=token_counter,
            tokenizer=tokenizer,
        )

    async def send_request(
        self,
        request: BenchmarkRequest,
        client: httpx.AsyncClient,
    ) -> Optional[dict[str, Any]]:
        """Send one real SSE request and return the historical result mapping."""

        result = await self.streaming_client.send_request(
            request.to_request_spec(self.model_name), client
        )
        self.metrics_tracker.record_result(result)
        if result.http_status == 429:
            self.metrics_tracker.record_preemption()
        if not result.success:
            logger.warning(
                "Request %s failed (%s): %s",
                request.request_id,
                result.error_type,
                result.error_message,
            )
            return None

        return {
            "request_id": request.request_id,
            "success": True,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "duration": result.e2e_ns / 1_000_000_000 if result.e2e_ns is not None else None,
            "ttft": result.ttft_ns / 1_000_000_000 if result.ttft_ns is not None else None,
            "tpot": result.tpot_ns / 1_000_000_000 if result.tpot_ns is not None else None,
            "itl": [value / 1_000_000_000 for value in result.itl_ns],
            "request_result": result.to_dict(),
        }


class BenchmarkRunner:
    """Historical runner facade using the new streaming scheduler and reducer."""

    def __init__(
        self,
        launcher: VLLMLauncher,
        concurrency: int = 10,
        warmup_requests: int = 5,
        max_tokens: int = 256,
    ) -> None:
        self.launcher = launcher
        self.concurrency = concurrency
        self.warmup_requests = warmup_requests
        self.max_tokens = max_tokens
        self.metrics_tracker = VLLMMetricsTracker()
        self.last_result: Optional[BenchmarkResult] = None

    async def run_benchmark(
        self,
        prompts: list[str],
        include_warmup: bool = True,
    ) -> dict[str, Any]:
        """Run a measured SSE benchmark while preserving the old return shape."""

        logger.info(
            "Starting benchmark with %s prompts, concurrency=%s",
            len(prompts),
            self.concurrency,
        )
        generator = RequestGenerator(prompts, max_tokens=self.max_tokens)
        client = SSEBenchmarkClient(
            self.launcher.base_url,
            model=self.launcher.config.model,
        )
        self.last_result = await client.run(
            generator.generate_specs(self.launcher.config.model),
            warmup_requests=self.warmup_requests if include_warmup else 0,
            max_concurrency=self.concurrency,
        )
        self.metrics_tracker.record_benchmark_result(self.last_result)
        summary = self.metrics_tracker.get_summary()
        logger.info(
            "Benchmark completed: %s/%s requests completed",
            summary["requests_completed"],
            summary["num_requests"],
        )
        return summary

    async def _run_warmup(self, prompts: list[str]) -> None:
        """Run standalone warmup requests without adding them to measurements."""

        if not prompts:
            return
        generator = RequestGenerator(prompts, max_tokens=self.max_tokens)
        client = SSEBenchmarkClient(
            self.launcher.base_url,
            model=self.launcher.config.model,
        )
        await client.run(
            generator.generate_specs(self.launcher.config.model),
            max_concurrency=self.concurrency,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get the most recent historical summary."""

        return self.metrics_tracker.get_summary()


class ResultCollector:
    """Collect and aggregate benchmark results across tuning trials."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def add_result(self, trial_id: str, result: dict[str, Any], params: dict[str, Any]) -> None:
        """Add a trial result."""

        self.results.append(
            {
                "trial_id": trial_id,
                "parameters": params,
                "metrics": result,
            }
        )

    def get_best(self, objective: str = "throughput_requests_per_sec") -> Optional[dict[str, Any]]:
        """Get the best result by objective."""

        if not self.results:
            return None
        return max(self.results, key=lambda value: value["metrics"].get(objective, 0.0))

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics across trials."""

        if not self.results:
            return {}
        throughputs = [
            result["metrics"].get("throughput_requests_per_sec", 0) for result in self.results
        ]
        latencies = [result["metrics"].get("avg_latency_ms", 0) for result in self.results]
        return {
            "num_trials": len(self.results),
            "throughput_mean": sum(throughputs) / len(throughputs),
            "throughput_max": max(throughputs),
            "throughput_min": min(throughputs),
            "latency_mean": sum(latencies) / len(latencies),
            "latency_min": min(latencies),
        }

    def to_dataframe(self) -> Optional[Any]:
        """Convert results to a pandas DataFrame when pandas is installed."""

        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("pandas not available, cannot create DataFrame")
            return None

        rows = []
        for result in self.results:
            row = {"trial_id": result["trial_id"]}
            row.update(result["parameters"])
            row.update(result["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)
