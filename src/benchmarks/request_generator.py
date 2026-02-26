"""Generate and execute benchmark requests against vLLM server."""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import httpx

from ..profiling.vllm_metrics import VLLMMetricsTracker
from ..vllm.launcher import VLLMLauncher

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRequest:
    """Container for a single benchmark request."""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0


class RequestGenerator:
    """Generate benchmark requests from prompts."""

    def __init__(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate_requests(self) -> List[BenchmarkRequest]:
        """Generate benchmark requests from prompts."""
        return [
            BenchmarkRequest(
                request_id=f"req_{i}",
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            for i, prompt in enumerate(self.prompts)
        ]

    def async_requests(self) -> AsyncIterator[BenchmarkRequest]:
        """Async iterator over requests."""
        for i, prompt in enumerate(self.prompts):
            yield BenchmarkRequest(
                request_id=f"req_{i}",
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )


class BenchmarkClient:
    """HTTP client for benchmarking vLLM server."""

    def __init__(
        self,
        base_url: str,
        metrics_tracker: VLLMMetricsTracker,
        model_name: str = "gpt2",
        timeout: int = 300,
    ):
        self.base_url = base_url
        self.metrics_tracker = metrics_tracker
        self.model_name = model_name
        self.timeout = timeout

    async def send_request(
        self,
        request: BenchmarkRequest,
        client: httpx.AsyncClient,
    ) -> Optional[Dict[str, Any]]:
        """Send a single completion request to vLLM."""
        url = f"{self.base_url}/v1/completions"

        self.metrics_tracker.record_request(request.request_id)

        start_time = time.time()
        ttft_recorded = False

        payload = {
            "model": self.model_name,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        try:
            async with client.stream("POST", url, json=payload, timeout=self.timeout) as response:
                response.raise_for_status()

                first_chunk_time = None
                output_tokens = 0

                async for chunk in response.aiter_bytes():
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttft = first_chunk_time - start_time
                        self.metrics_tracker.record_ttft(request.request_id, ttft)
                        ttft_recorded = True

                    try:
                        chunk_str = chunk.decode("utf-8")
                        if "choices" in chunk_str:
                            import json

                            try:
                                data = json.loads(chunk_str)
                                if "choices" in data and data["choices"]:
                                    text = data["choices"][0].get("text", "")
                                    if text:
                                        output_tokens += 1
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        pass

                completion_time = time.time()
                self.metrics_tracker.record_completion(request.request_id, output_tokens)

                return {
                    "request_id": request.request_id,
                    "success": True,
                    "output_tokens": output_tokens,
                    "duration": completion_time - start_time,
                }

        except httpx.TimeoutException:
            logger.warning(f"Request {request.request_id} timed out")
            self.metrics_tracker.record_error("timeout")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Request {request.request_id} failed: {e.response.status_code}")
            self.metrics_tracker.record_error("http")

            if e.response.status_code == 429:
                self.metrics_tracker.record_preemption()

            return None

        except Exception as e:
            logger.error(f"Request {request.request_id} error: {e}")
            self.metrics_tracker.record_error("general")
            return None


class BenchmarkRunner:
    """Run benchmark with concurrent requests."""

    def __init__(
        self,
        launcher: VLLMLauncher,
        concurrency: int = 10,
        warmup_requests: int = 5,
        max_tokens: int = 256,
    ):
        self.launcher = launcher
        self.concurrency = concurrency
        self.warmup_requests = warmup_requests
        self.max_tokens = max_tokens
        self.metrics_tracker = VLLMMetricsTracker()

    async def run_benchmark(
        self,
        prompts: List[str],
        include_warmup: bool = True,
    ) -> Dict[str, Any]:
        """Run benchmark against vLLM server."""
        logger.info(
            f"Starting benchmark with {len(prompts)} prompts, concurrency={self.concurrency}"
        )

        if include_warmup and self.warmup_requests > 0:
            await self._run_warmup(prompts[: self.warmup_requests])

        self.metrics_tracker.start_benchmark()

        generator = RequestGenerator(prompts, max_tokens=self.max_tokens)
        requests = generator.generate_requests()

        client = BenchmarkClient(
            self.launcher.base_url,
            self.metrics_tracker,
            model_name=self.launcher.config.model,
        )

        async with httpx.AsyncClient(timeout=300) as http_client:
            semaphore = asyncio.Semaphore(self.concurrency)

            async def execute_request(req: BenchmarkRequest) -> Optional[Dict[str, Any]]:
                async with semaphore:
                    return await client.send_request(req, http_client)

            tasks = [execute_request(req) for req in requests]
            await asyncio.gather(*tasks, return_exceptions=True)

        self.metrics_tracker.end_benchmark()

        summary = self.metrics_tracker.get_summary()
        logger.info(
            f"Benchmark completed: {summary['requests_completed']}/{summary['num_requests']} requests completed"
        )

        return summary

    async def _run_warmup(self, prompts: List[str]) -> None:
        """Run warmup requests."""
        if not prompts:
            return

        logger.info(f"Running {len(prompts)} warmup requests...")

        warmup_metrics = VLLMMetricsTracker()
        warmup_metrics.start_benchmark()

        generator = RequestGenerator(prompts, max_tokens=self.max_tokens)
        requests = generator.generate_requests()

        client = BenchmarkClient(
            self.launcher.base_url,
            warmup_metrics,
            model_name=self.launcher.config.model,
        )

        async with httpx.AsyncClient(timeout=300) as http_client:
            for req in requests:
                await client.send_request(req, http_client)

        warmup_metrics.end_benchmark()
        logger.info("Warmup completed")

    def get_metrics(self) -> Dict[str, Any]:
        """Get benchmark metrics."""
        return self.metrics_tracker.get_summary()


class ResultCollector:
    """Collect and aggregate benchmark results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add_result(self, trial_id: str, result: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Add a trial result."""
        self.results.append(
            {
                "trial_id": trial_id,
                "parameters": params,
                "metrics": result,
            }
        )

    def get_best(self, objective: str = "throughput_requests_per_sec") -> Optional[Dict[str, Any]]:
        """Get the best result by objective."""
        if not self.results:
            return None

        best = max(self.results, key=lambda x: x["metrics"].get(objective, 0.0))
        return best

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results."""
        if not self.results:
            return {}

        throughputs = [r["metrics"].get("throughput_requests_per_sec", 0) for r in self.results]
        latencies = [r["metrics"].get("avg_latency_ms", 0) for r in self.results]

        return {
            "num_trials": len(self.results),
            "throughput_mean": sum(throughputs) / len(throughputs) if throughputs else 0,
            "throughput_max": max(throughputs) if throughputs else 0,
            "throughput_min": min(throughputs) if throughputs else 0,
            "latency_mean": sum(latencies) / len(latencies) if latencies else 0,
            "latency_min": min(latencies) if latencies else 0,
        }

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd

            rows = []
            for r in self.results:
                row = {"trial_id": r["trial_id"]}
                row.update(r["parameters"])
                row.update(r["metrics"])
                rows.append(row)

            return pd.DataFrame(rows)

        except ImportError:
            logger.warning("pandas not available, cannot create DataFrame")
            return None
