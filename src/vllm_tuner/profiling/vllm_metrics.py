"""Compatibility metrics facade backed by typed benchmark request results."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np

from vllm_tuner.benchmarks.metrics import aggregate_request_results
from vllm_tuner.benchmarks.models import (
    BenchmarkResult,
    RequestResult,
    RequestStatus,
)

logger = logging.getLogger(__name__)


class VLLMMetrics:
    """Legacy-shaped metrics populated from trustworthy request measurements."""

    def __init__(self) -> None:
        self.reset()

    def to_dict(self) -> dict[str, Any]:
        """Return canonical metrics plus the historical compatibility keys."""

        aggregate = aggregate_request_results(
            self.request_results,
            started_at=self.start_ns,
            finished_at=self.end_ns,
            percentiles=(50.0, 95.0, 99.0),
        )
        duration = aggregate["duration"]
        avg_prefill = self._mean(self.prefill_times)
        avg_decode = self._mean(self.decode_times)

        # For callers that populated the container manually, retain their
        # counters. RequestResult-derived values remain authoritative whenever
        # raw request results exist.
        has_results = bool(self.request_results)
        num_requests = (
            self._integer(aggregate["num_requests"]) if has_results else self.num_requests
        )
        completed = (
            self._integer(aggregate["completed"]) if has_results else self.requests_completed
        )
        errors = self._integer(aggregate["failed"]) if has_results else self.errors
        input_tokens = (
            self._integer(aggregate["total_input_tokens"]) if has_results else self.input_tokens
        )
        output_tokens = (
            self._integer(aggregate["total_output_tokens"]) if has_results else self.output_tokens
        )
        duration_seconds = float(duration) if isinstance(duration, (int, float)) else None
        avg_ttft_ms = (
            aggregate["mean_ttft_ms"] if has_results else self._mean(self.ttft_times) * 1000
        )
        avg_tpot_ms = (
            aggregate["mean_tpot_ms"] if has_results else self._mean(self.tpot_times) * 1000
        )
        avg_e2e_ms = aggregate["mean_e2e_ms"] if has_results else self._mean(self.e2e_times) * 1000
        p50_e2e_ms = (
            aggregate["p50_e2e_ms"] if has_results else self._percentile(self.e2e_times, 50) * 1000
        )
        p95_e2e_ms = (
            aggregate["p95_e2e_ms"] if has_results else self._percentile(self.e2e_times, 95) * 1000
        )
        p99_e2e_ms = (
            aggregate["p99_e2e_ms"] if has_results else self._percentile(self.e2e_times, 99) * 1000
        )
        request_throughput = aggregate["request_throughput"]
        output_throughput = aggregate["output_throughput"]
        if not has_results and duration_seconds and duration_seconds > 0:
            request_throughput = completed / duration_seconds
            output_throughput = output_tokens / duration_seconds

        result = {
            **aggregate,
            "prefill_times": list(self.prefill_times),
            "decode_times": list(self.decode_times),
            "ttft_times": list(self.ttft_times),
            "tpot_times": list(self.tpot_times),
            "itl_times": list(self.itl_times),
            "e2e_times": list(self.e2e_times),
            "preemption_counts": self.preemption_counts,
            "num_requests": num_requests,
            "input_tokens": input_tokens,
            "total_tokens": input_tokens + output_tokens,
            "output_tokens": output_tokens,
            "requests_processed": self.requests_processed,
            "requests_completed": completed,
            "errors": errors,
            "oom_errors": self.oom_errors,
            "avg_prefill_time_ms": avg_prefill * 1000,
            "avg_decode_time_ms": avg_decode * 1000,
            "avg_ttft_ms": avg_ttft_ms,
            "avg_tpot_ms": avg_tpot_ms,
            # Historical "latency" fields now mean independently measured E2E.
            "avg_latency_ms": avg_e2e_ms,
            "p50_latency_ms": p50_e2e_ms,
            "p95_latency_ms": p95_e2e_ms,
            "p99_latency_ms": p99_e2e_ms,
            "throughput_requests_per_sec": request_throughput,
            "throughput_tokens_per_sec": output_throughput,
            "duration_seconds": duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }
        return result

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return int(value)
        return 0

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Compatibility helper using NumPy interpolation."""

        if not data:
            return 0.0
        return float(np.percentile(np.asarray(data, dtype=np.float64), percentile))

    def reset(self) -> None:
        """Reset all metrics and raw per-request measurements."""

        self.prefill_times: list[float] = []
        self.decode_times: list[float] = []
        self.ttft_times: list[float] = []
        self.tpot_times: list[float] = []
        self.itl_times: list[float] = []
        self.e2e_times: list[float] = []
        self.request_results: list[RequestResult] = []
        self.preemption_counts = 0
        self.num_requests = 0
        self.input_tokens = 0
        self.total_tokens = 0
        self.output_tokens = 0
        self.requests_processed = 0
        self.requests_completed = 0
        self.errors = 0
        self.oom_errors = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.start_ns: Optional[int] = None
        self.end_ns: Optional[int] = None


class VLLMMetricsTracker:
    """Adapt old record-style calls to the new typed benchmark metrics core."""

    def __init__(self) -> None:
        self.metrics = VLLMMetrics()
        self._request_start_times: dict[str, int] = {}
        self._request_first_token: dict[str, Optional[int]] = {}
        self._known_request_ids: set[str] = set()

    def start_benchmark(self) -> None:
        """Start a measurement window using a monotonic nanosecond clock."""

        self.metrics.reset()
        self._request_start_times.clear()
        self._request_first_token.clear()
        self._known_request_ids.clear()
        self.metrics.start_time = datetime.now()
        self.metrics.start_ns = time.perf_counter_ns()
        logger.debug("Benchmark started")

    def end_benchmark(self) -> None:
        """End the current monotonic measurement window."""

        self.metrics.end_ns = time.perf_counter_ns()
        self.metrics.end_time = datetime.now()
        duration = (
            (self.metrics.end_ns - self.metrics.start_ns) / 1_000_000_000
            if self.metrics.start_ns is not None
            else None
        )
        logger.debug("Benchmark ended, duration=%s seconds", duration)

    def record_request(self, request_id: str) -> None:
        """Record a legacy request start with ``perf_counter_ns``."""

        if request_id not in self._known_request_ids:
            self.metrics.num_requests += 1
            self._known_request_ids.add(request_id)
        self._request_start_times[request_id] = time.perf_counter_ns()
        self._request_first_token[request_id] = None

    def record_ttft(self, request_id: str, ttft: float) -> None:
        """Record a legacy TTFT value in seconds without using wall-clock time."""

        start = self._request_start_times.get(request_id)
        if start is None:
            return
        first_token = start + max(0, int(round(ttft * 1_000_000_000)))
        self._request_first_token[request_id] = first_token

    def record_completion(self, request_id: str, output_tokens: int, input_tokens: int = 0) -> None:
        """Finish a legacy request and convert it to a typed raw result."""

        start = self._request_start_times.get(request_id)
        if start is None:
            return
        first_token = self._request_first_token.get(request_id)
        finished = time.perf_counter_ns()
        result = RequestResult(
            request_id=request_id,
            scheduled_at=start,
            sent_at=start,
            first_token_at=first_token,
            finished_at=finished,
            input_tokens=input_tokens,
            output_tokens=max(0, output_tokens),
            token_timestamps=[first_token] if first_token is not None else [],
            status=RequestStatus.SUCCESS,
            token_count_source="legacy_record_api",
        )
        self.record_result(result)

    def record_result(self, result: RequestResult) -> None:
        """Record one authoritative result from the streaming benchmark core."""

        if result.warmup:
            return
        if result.request_id not in self._known_request_ids:
            self._known_request_ids.add(result.request_id)
            self.metrics.num_requests += 1

        self.metrics.request_results.append(result)
        self.metrics.input_tokens += result.input_tokens if result.success else 0
        self.metrics.output_tokens += result.output_tokens if result.success else 0
        self.metrics.total_tokens = self.metrics.input_tokens + self.metrics.output_tokens

        if result.success:
            self.metrics.requests_completed += 1
            if result.first_token_at is not None:
                self.metrics.requests_processed += 1
            if result.ttft_ns is not None:
                self.metrics.ttft_times.append(result.ttft_ns / 1_000_000_000)
            if result.tpot_ns is not None:
                self.metrics.tpot_times.append(result.tpot_ns / 1_000_000_000)
            if result.e2e_ns is not None:
                self.metrics.e2e_times.append(result.e2e_ns / 1_000_000_000)
            self.metrics.itl_times.extend(value / 1_000_000_000 for value in result.itl_ns)
        else:
            self.metrics.errors += 1
            if result.error_type == "oom":
                self.metrics.oom_errors += 1

        self._request_start_times.pop(result.request_id, None)
        self._request_first_token.pop(result.request_id, None)

    def record_benchmark_result(self, result: BenchmarkResult) -> None:
        """Replace tracker state with a complete new-core benchmark result."""

        self.metrics.reset()
        self._request_start_times.clear()
        self._request_first_token.clear()
        self._known_request_ids.clear()
        self.metrics.start_ns = result.started_at
        self.metrics.end_ns = result.finished_at
        self.metrics.start_time = datetime.now()
        for request_result in result.request_results:
            self.record_result(request_result)
        self.metrics.end_time = datetime.now()

    load_benchmark_result = record_benchmark_result

    def record_preemption(self) -> None:
        """Record a client-observed preemption diagnostic."""

        self.metrics.preemption_counts += 1
        logger.debug("Preemption recorded")

    def record_error(self, error_type: str = "general") -> None:
        """Record an unassociated legacy error diagnostic."""

        self.metrics.errors += 1
        if error_type == "oom":
            self.metrics.oom_errors += 1
        logger.debug("Error recorded: %s", error_type)

    def disable_prefilling(self) -> None:
        """Track a legacy prefill throttle diagnostic."""

        self.record_preemption()

    def get_metrics(self) -> VLLMMetrics:
        """Get the compatibility metrics container."""

        return self.metrics

    def get_summary(self) -> dict[str, Any]:
        """Get canonical and legacy-compatible aggregate metrics."""

        return self.metrics.to_dict()
