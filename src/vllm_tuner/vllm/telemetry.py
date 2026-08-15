"""Compatibility telemetry facade with Prometheus as the primary data source."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from vllm_tuner.profiling.prometheus import (
    PrometheusCollector,
    PrometheusSnapshot,
    parse_vllm_metrics,
)
from vllm_tuner.profiling.session import TelemetrySession

logger = logging.getLogger(__name__)


class VLLMTelemetryParser:
    """Compatibility parser for engine telemetry.

    Prometheus exposition is authoritative. Regex log parsing remains available
    only for startup/error diagnostics and labels every result as a non-
    authoritative fallback so it cannot be mistaken for measurement data.
    """

    PATTERNS = {
        "init_block_manager": re.compile(
            r"Initialized a KV cache with initial memory capacity of ([\d.]+) (GB|MB|TB)"
        ),
        "block_manager_stats": re.compile(
            r"Block manager stats: CPU cache hit: ([\d.]+) %, CPU cache miss: ([\d.]+) %"
        ),
        "cpu_swap": re.compile(
            r"Swap out ([\d]+) requests to CPU, Swap in ([\d]+) requests from CPU"
        ),
        "prefill_throttle": re.compile(
            r"Prefill stage: throttled ([\d]+) requests due to limited memory"
        ),
        "decode_per_token_latency": re.compile(r"decode time per token: ([\d.]+) ms"),
        "prefill_latency": re.compile(r"prefill time per token: ([\d.]+) ms"),
        "throughput": re.compile(r"Throughput: ([\d.]+) tokens/sec"),
        "runtime_error": re.compile(r"RuntimeError"),
        "oom_error": re.compile(r"(CUDA out of memory|OutOfMemoryError|OOMError)", re.IGNORECASE),
    }

    def __init__(self) -> None:
        self.metrics = self._empty_metrics()
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        # Numeric defaults are retained for legacy display code. ``available``
        # and ``authoritative`` prevent these defaults from posing as samples.
        return {
            "available": False,
            "authoritative": False,
            "source": "unavailable",
            "primary_source": "prometheus",
            "fallback_reason": None,
            "engine": {},
            "kv_cache_utilization": 0.0,
            "slot_occupancy": 0.0,
            "preemption_count": 0,
            "cpu_cache_hit_rate": 0.0,
            "cpu_cache_miss_rate": 0.0,
            "prefill_throttled_count": 0,
            "prefill_latency_ms": 0.0,
            "decode_latency_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "max_memory_gb": 0.0,
            "swap_out_count": 0,
            "swap_in_count": 0,
            "runtime_error_detected": False,
            "oom_detected": False,
        }

    def parse_prometheus_content(self, content: str) -> dict[str, Any]:
        """Parse authoritative vLLM ``/metrics`` exposition."""

        engine = parse_vllm_metrics(content)
        self.metrics = self._empty_metrics()
        self.events = []
        self.metrics.update(
            {
                "available": True,
                "authoritative": True,
                "source": "prometheus",
                "measurement_scope": "snapshot",
                "engine": engine,
                "kv_cache_utilization": self._optional_number(engine.get("kv_cache_usage_perc")),
                "slot_occupancy": self._optional_number(engine.get("num_requests_running")),
                "preemption_count": self._optional_number(engine.get("num_preemptions_total")),
                "prefill_latency_ms": None,
                "decode_latency_ms": self._histogram_mean_ms(
                    engine.get("inter_token_latency_seconds")
                ),
                "throughput_tokens_per_sec": None,
            }
        )
        return self.metrics

    parse_metrics_content = parse_prometheus_content

    async def collect(self, collector: PrometheusCollector) -> dict[str, Any]:
        """Collect one authoritative Prometheus snapshot through the new core."""

        snapshot = await collector.collect()
        return self.parse_prometheus_snapshot(snapshot)

    def parse_prometheus_snapshot(self, snapshot: PrometheusSnapshot) -> dict[str, Any]:
        """Consume a typed Prometheus snapshot."""

        if not snapshot.endpoint_available:
            self.metrics = self._empty_metrics()
            self.events = []
            self.metrics.update(
                {
                    "source": "prometheus_unavailable",
                    "fallback_reason": snapshot.error,
                    "engine": snapshot.metrics,
                }
            )
            return self.metrics
        self.metrics = self._empty_metrics()
        self.events = []
        engine = snapshot.metrics
        self.metrics.update(
            {
                "available": True,
                "authoritative": True,
                "source": "prometheus",
                "measurement_scope": "snapshot",
                "engine": engine,
                "kv_cache_utilization": self._optional_number(engine.get("kv_cache_usage_perc")),
                "slot_occupancy": self._optional_number(engine.get("num_requests_running")),
                "preemption_count": self._optional_number(engine.get("num_preemptions_total")),
                "prefill_latency_ms": None,
                "decode_latency_ms": self._histogram_mean_ms(
                    engine.get("inter_token_latency_seconds")
                ),
                "throughput_tokens_per_sec": None,
            }
        )
        return self.metrics

    def parse_log_file(self, log_path: Path) -> dict[str, Any]:
        """Parse logs only as an explicitly labelled diagnostic fallback."""

        if not log_path.exists():
            logger.warning("Diagnostic log file not found: %s", log_path)
            self.metrics = self._empty_metrics()
            self.events = []
            self.metrics["source"] = "log_diagnostic_fallback"
            self.metrics["fallback_reason"] = "log file not found"
            return self.metrics
        return self.parse_log_content(log_path.read_text(encoding="utf-8"))

    def parse_log_content(self, log_content: str) -> dict[str, Any]:
        """Parse unstable log text for diagnostics, never primary metrics."""

        self.metrics = self._empty_metrics()
        self.events = []
        self.metrics["available"] = bool(log_content.strip())
        self.metrics["authoritative"] = False
        self.metrics["source"] = "log_diagnostic_fallback"
        self.metrics["fallback_reason"] = "Prometheus data was not supplied"
        for line in log_content.splitlines():
            self._parse_line(line)

        cache_total = self.metrics["cpu_cache_hit_rate"] + self.metrics["cpu_cache_miss_rate"]
        if cache_total > 0:
            self.metrics["cpu_cache_hit_rate"] /= cache_total
            self.metrics["cpu_cache_miss_rate"] /= cache_total
        return self.metrics

    def _parse_line(self, line: str) -> None:
        timestamp = self._extract_timestamp(line)

        match = self.PATTERNS["init_block_manager"].search(line)
        if match:
            value = float(match.group(1))
            self.metrics["max_memory_gb"] = self._convert_to_gb(value, match.group(2))
            self.events.append(
                {
                    "type": "kv_cache_init",
                    "value_gb": self.metrics["max_memory_gb"],
                    "timestamp": timestamp,
                    "source": "log_diagnostic_fallback",
                }
            )

        match = self.PATTERNS["block_manager_stats"].search(line)
        if match:
            self.metrics["cpu_cache_hit_rate"] = float(match.group(1))
            self.metrics["cpu_cache_miss_rate"] = float(match.group(2))

        match = self.PATTERNS["cpu_swap"].search(line)
        if match:
            self.metrics["swap_out_count"] = int(match.group(1))
            self.metrics["swap_in_count"] = int(match.group(2))

        match = self.PATTERNS["prefill_throttle"].search(line)
        if match:
            count = int(match.group(1))
            self.metrics["prefill_throttled_count"] = count
            self.metrics["preemption_count"] += count

        match = self.PATTERNS["decode_per_token_latency"].search(line)
        if match:
            self.metrics["decode_latency_ms"] = float(match.group(1))

        match = self.PATTERNS["prefill_latency"].search(line)
        if match:
            self.metrics["prefill_latency_ms"] = float(match.group(1))

        match = self.PATTERNS["throughput"].search(line)
        if match:
            self.metrics["throughput_tokens_per_sec"] = float(match.group(1))

        if self.PATTERNS["runtime_error"].search(line):
            self.metrics["runtime_error_detected"] = True
        oom_match = self.PATTERNS["oom_error"].search(line)
        if oom_match:
            self.metrics["oom_detected"] = True
            self.events.append(
                {
                    "type": "error",
                    "message": oom_match.group(1),
                    "timestamp": timestamp,
                    "source": "log_diagnostic_fallback",
                }
            )

    @staticmethod
    def _optional_number(value: object) -> Optional[float]:
        return float(value) if isinstance(value, (int, float)) else None

    @classmethod
    def _histogram_mean_ms(cls, value: object) -> Optional[float]:
        if not isinstance(value, dict):
            return None
        mean = cls._optional_number(value.get("mean"))
        return mean * 1000 if mean is not None else None

    @staticmethod
    def _extract_timestamp(line: str) -> Optional[datetime]:
        timestamp_patterns = [
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
            r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\]",
        ]
        for pattern in timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    return datetime.fromisoformat(match.group(0).replace("[", "").replace("]", ""))
                except ValueError:
                    pass
        return None

    @staticmethod
    def _convert_to_gb(value: float, unit: str) -> float:
        lowered = unit.lower()
        if lowered == "mb":
            return value / 1024
        if lowered == "tb":
            return value * 1024
        return value

    def get_summary(self) -> dict[str, Any]:
        """Get metrics and fallback provenance."""

        return {**self.metrics, "events_count": len(self.events)}

    def reset(self) -> None:
        """Reset metrics and diagnostic events."""

        self.metrics = self._empty_metrics()
        self.events = []


def parse_vllm_prometheus(content: str) -> dict[str, Any]:
    """Parse the authoritative vLLM Prometheus endpoint payload."""

    return VLLMTelemetryParser().parse_prometheus_content(content)


def parse_vllm_logs(log_path: Path) -> dict[str, Any]:
    """Parse a diagnostic log fallback with explicit provenance fields."""

    return VLLMTelemetryParser().parse_log_file(log_path)


def detect_oom_from_logs(log_path: Path) -> bool:
    """Diagnose OOM text when Prometheus cannot expose process failures."""

    parser = VLLMTelemetryParser()
    parser.parse_log_file(log_path)
    return bool(parser.metrics["oom_detected"])


# Old import locations now resolve to the measurement-window core.
VLLMTelemetrySession = TelemetrySession

__all__ = [
    "PrometheusCollector",
    "TelemetrySession",
    "VLLMTelemetryParser",
    "VLLMTelemetrySession",
    "detect_oom_from_logs",
    "parse_vllm_logs",
    "parse_vllm_prometheus",
]
