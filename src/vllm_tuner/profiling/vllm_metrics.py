"""Collect vLLM-specific metrics during benchmarking."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VLLMMetrics:
    """Container for vLLM performance metrics."""

    def __init__(self):
        self.prefill_times: list[float] = []
        self.decode_times: list[float] = []
        self.ttft_times: list[float] = []
        self.tpot_times: list[float] = []
        self.preemption_counts: int = 0
        self.num_requests: int = 0
        self.total_tokens: int = 0
        self.output_tokens: int = 0
        self.requests_processed: int = 0
        self.requests_completed: int = 0
        self.errors: int = 0
        self.oom_errors: int = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        duration_seconds = 0.0
        if self.start_time and self.end_time:
            duration_seconds = (self.end_time - self.start_time).total_seconds()

        avg_prefill = (
            sum(self.prefill_times) / len(self.prefill_times) if self.prefill_times else 0.0
        )
        avg_decode = sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0.0
        avg_ttft = sum(self.ttft_times) / len(self.ttft_times) if self.ttft_times else 0.0
        avg_tpot = sum(self.tpot_times) / len(self.tpot_times) if self.tpot_times else 0.0

        latencies = []
        avg_latency = 0.0
        if self.ttft_times and self.tpot_times:
            latencies = [ttft + tpot for ttft, tpot in zip(self.ttft_times, self.tpot_times)]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)

        throughput = 0.0
        if duration_seconds > 0 and self.requests_completed > 0:
            throughput = self.requests_completed / duration_seconds

        tokens_per_sec = 0.0
        if duration_seconds > 0 and self.output_tokens > 0:
            tokens_per_sec = self.output_tokens / duration_seconds

        p50_latency = self._percentile(latencies, 50) if latencies else 0.0
        p95_latency = self._percentile(latencies, 95) if latencies else 0.0
        p99_latency = self._percentile(latencies, 99) if latencies else 0.0

        return {
            "prefill_times": self.prefill_times,
            "decode_times": self.decode_times,
            "ttft_times": self.ttft_times,
            "tpot_times": self.tpot_times,
            "preemption_counts": self.preemption_counts,
            "num_requests": self.num_requests,
            "total_tokens": self.total_tokens,
            "output_tokens": self.output_tokens,
            "requests_processed": self.requests_processed,
            "requests_completed": self.requests_completed,
            "errors": self.errors,
            "oom_errors": self.oom_errors,
            "avg_prefill_time_ms": avg_prefill * 1000,
            "avg_decode_time_ms": avg_decode * 1000,
            "avg_ttft_ms": avg_ttft * 1000,
            "avg_tpot_ms": avg_tpot * 1000,
            "avg_latency_ms": avg_latency * 1000,
            "p50_latency_ms": p50_latency * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "p99_latency_ms": p99_latency * 1000,
            "throughput_requests_per_sec": throughput,
            "throughput_tokens_per_sec": tokens_per_sec,
            "duration_seconds": duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def reset(self) -> None:
        """Reset all metrics."""
        self.prefill_times = []
        self.decode_times = []
        self.ttft_times = []
        self.tpot_times = []
        self.preemption_counts = 0
        self.num_requests = 0
        self.total_tokens = 0
        self.output_tokens = 0
        self.requests_processed = 0
        self.requests_completed = 0
        self.errors = 0
        self.oom_errors = 0
        self.start_time = None
        self.end_time = None


class VLLMMetricsTracker:
    """Track and compute vLLM metrics during benchmark runs."""

    def __init__(self):
        self.metrics = VLLMMetrics()
        self._request_start_times: Dict[str, datetime] = {}
        self._request_ttft: Dict[str, Optional[float]] = {}

    def start_benchmark(self) -> None:
        """Start benchmark run."""
        self.metrics.reset()
        self.metrics.start_time = datetime.now()
        logger.debug("Benchmark started")

    def end_benchmark(self) -> None:
        """End benchmark run."""
        self.metrics.end_time = datetime.now()
        if self.metrics.start_time:
            duration = (self.metrics.end_time - self.metrics.start_time).total_seconds()
            logger.debug(f"Benchmark ended, duration: {duration:.2f}s")
        else:
            logger.debug("Benchmark ended (no start time recorded)")

    def record_request(self, request_id: str) -> None:
        """Record a new request."""
        self.metrics.num_requests += 1
        self._request_start_times[request_id] = datetime.now()
        self._request_ttft[request_id] = None

    def record_ttft(self, request_id: str, ttft: float) -> None:
        """Record Time To First Token for a request."""
        if request_id in self._request_ttft:
            self._request_ttft[request_id] = ttft
            self.metrics.ttft_times.append(ttft)
            self.metrics.requests_processed += 1

    def record_completion(self, request_id: str, output_tokens: int) -> None:
        """Record request completion."""
        if request_id not in self._request_start_times:
            return

        start_time = self._request_start_times[request_id]
        total_time = (datetime.now() - start_time).total_seconds()

        if request_id in self._request_ttft:
            ttft = self._request_ttft[request_id]
            if ttft is not None:
                if output_tokens > 0:
                    tpot = (total_time - ttft) / output_tokens
                    self.metrics.tpot_times.append(tpot)

        self.metrics.output_tokens += output_tokens
        self.metrics.requests_completed += 1

        del self._request_start_times[request_id]
        if request_id in self._request_ttft:
            del self._request_ttft[request_id]

    def record_preemption(self) -> None:
        """Record a preemption event."""
        self.metrics.preemption_counts += 1
        logger.debug("Preemption recorded")

    def record_error(self, error_type: str = "general") -> None:
        """Record an error."""
        self.metrics.errors += 1
        if error_type == "oom":
            self.metrics.oom_errors += 1
        logger.debug(f"Error recorded: {error_type}")

    def disable_prefilling(self) -> None:
        """Track prefill throttle/prevention."""
        self.record_preemption()

    def get_metrics(self) -> VLLMMetrics:
        """Get current metrics."""
        return self.metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics.to_dict()
