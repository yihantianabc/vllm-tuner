"""Parse vLLM logs and extract telemetry metrics."""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class VLLMTelemetryParser:
    """Parse vLLM logs to extract telemetry metrics."""

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
        "runtime_error": re.compile(r"(RuntimeError|CUDA out of memory|OOMError)"),
    }

    def __init__(self):
        self.metrics: Dict[str, Any] = {
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
            "oom_detected": False,
        }
        self.events: list[Dict[str, Any]] = []

    def parse_log_file(self, log_path: Path) -> Dict[str, Any]:
        """Parse vLLM log file and extract metrics."""
        if not log_path.exists():
            logger.warning(f"Log file not found: {log_path}")
            return self.metrics

        with open(log_path, "r") as f:
            log_content = f.read()

        return self.parse_log_content(log_content)

    def parse_log_content(self, log_content: str) -> Dict[str, Any]:
        """Parse log content string."""
        lines = log_content.split("\n")

        for line in lines:
            self._parse_line(line)

        if self.metrics["cpu_cache_hit_rate"] + self.metrics["cpu_cache_miss_rate"] > 0:
            total_cache_ops = (
                self.metrics["cpu_cache_hit_rate"] + self.metrics["cpu_cache_miss_rate"]
            )
            if total_cache_ops > 0:
                self.metrics["cpu_cache_hit_rate"] /= total_cache_ops * 100
                self.metrics["cpu_cache_miss_rate"] /= total_cache_ops * 100

        return self.metrics

    def _parse_line(self, line: str) -> None:
        """Parse individual log line."""
        timestamp = self._extract_timestamp(line)

        match = self.PATTERNS["init_block_manager"].search(line)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            self.metrics["max_memory_gb"] = self._convert_to_gb(value, unit)
            self.events.append(
                {
                    "type": "kv_cache_init",
                    "value_gb": self.metrics["max_memory_gb"],
                    "timestamp": timestamp,
                }
            )

        match = self.PATTERNS["block_manager_stats"].search(line)
        if match:
            hit_rate = float(match.group(1))
            miss_rate = float(match.group(2))
            self.metrics["cpu_cache_hit_rate"] = hit_rate
            self.metrics["cpu_cache_miss_rate"] = miss_rate

        match = self.PATTERNS["cpu_swap"].search(line)
        if match:
            swap_out = int(match.group(1))
            swap_in = int(match.group(2))
            self.metrics["swap_out_count"] = swap_out
            self.metrics["swap_in_count"] = swap_in

        match = self.PATTERNS["prefill_throttle"].search(line)
        if match:
            count = int(match.group(1))
            self.metrics["prefill_throttled_count"] = count
            self.metrics["preemption_count"] += count

        match = self.PATTERNS["decode_per_token_latency"].search(line)
        if match:
            latency = float(match.group(1))
            self.metrics["decode_latency_ms"] = latency

        match = self.PATTERNS["prefill_latency"].search(line)
        if match:
            latency = float(match.group(1))
            self.metrics["prefill_latency_ms"] = latency

        match = self.PATTERNS["throughput"].search(line)
        if match:
            throughput = float(match.group(1))
            self.metrics["throughput_tokens_per_sec"] = throughput

        match = self.PATTERNS["runtime_error"].search(line)
        if match:
            self.metrics["oom_detected"] = True
            self.events.append(
                {
                    "type": "error",
                    "message": match.group(1),
                    "timestamp": timestamp,
                }
            )

    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line if present."""
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

    def _convert_to_gb(self, value: float, unit: str) -> float:
        """Convert memory value to GB."""
        unit = unit.lower()
        if unit == "gb":
            return value
        elif unit == "mb":
            return value / 1024
        elif unit == "tb":
            return value * 1024
        else:
            return value

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of parsed metrics."""
        summary = self.metrics.copy()
        summary["events_count"] = len(self.events)
        return summary

    def reset(self) -> None:
        """Reset metrics and events."""
        self.metrics = {
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
            "oom_detected": False,
        }
        self.events = []


def parse_vllm_logs(log_path: Path) -> Dict[str, Any]:
    """Convenience function to parse vLLM logs."""
    parser = VLLMTelemetryParser()
    return parser.parse_log_file(log_path)


def detect_oom_from_logs(log_path: Path) -> bool:
    """Check if OOM error occurred in logs."""
    parser = VLLMTelemetryParser()
    parser.parse_log_file(log_path)
    return parser.metrics["oom_detected"]
