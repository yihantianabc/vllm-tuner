"""Prometheus exposition parsing and vLLM engine telemetry collection."""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import httpx

from .timeseries import (
    SampleTimestamp,
    capture_timestamp,
    counter_window_delta,
    summarize_values,
)

# Metric names changed between vLLM releases and between the legacy and
# multiprocess Prometheus exporters. Canonical keys remain stable in artifacts.
VLLM_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "num_requests_running": (
        "vllm:num_requests_running",
        "vllm_num_requests_running",
        "vllm:num_requests_running_gpu",
        "vllm_num_requests_running_gpu",
    ),
    "num_requests_waiting": (
        "vllm:num_requests_waiting",
        "vllm_num_requests_waiting",
        "vllm:num_requests_waiting_gpu",
        "vllm_num_requests_waiting_gpu",
    ),
    "kv_cache_usage_perc": (
        "vllm:kv_cache_usage_perc",
        "vllm_kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
        "vllm:gpu_cache_usage_percent",
        "vllm_gpu_cache_usage_percent",
    ),
    "num_preemptions_total": (
        "vllm:num_preemptions_total",
        "vllm_num_preemptions_total",
        "vllm:num_preemptions",
        "vllm_num_preemptions",
    ),
    "prompt_tokens_total": (
        "vllm:prompt_tokens_total",
        "vllm_prompt_tokens_total",
    ),
    "generation_tokens_total": (
        "vllm:generation_tokens_total",
        "vllm_generation_tokens_total",
    ),
    "prefix_cache_queries": (
        "vllm:prefix_cache_queries",
        "vllm_prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
        "vllm_prefix_cache_queries_total",
    ),
    "prefix_cache_hits": (
        "vllm:prefix_cache_hits",
        "vllm_prefix_cache_hits",
        "vllm:prefix_cache_hits_total",
        "vllm_prefix_cache_hits_total",
    ),
    "time_to_first_token_seconds": (
        "vllm:time_to_first_token_seconds",
        "vllm_time_to_first_token_seconds",
    ),
    "inter_token_latency_seconds": (
        "vllm:inter_token_latency_seconds",
        "vllm_inter_token_latency_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm_time_per_output_token_seconds",
    ),
    "e2e_request_latency_seconds": (
        "vllm:e2e_request_latency_seconds",
        "vllm_e2e_request_latency_seconds",
    ),
    "request_queue_time_seconds": (
        "vllm:request_queue_time_seconds",
        "vllm_request_queue_time_seconds",
        "vllm:queue_time_seconds",
        "vllm_queue_time_seconds",
    ),
}

GAUGE_METRICS = (
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
)
COUNTER_METRICS = (
    "num_preemptions_total",
    "prompt_tokens_total",
    "generation_tokens_total",
    "prefix_cache_queries",
    "prefix_cache_hits",
)
HISTOGRAM_METRICS = (
    "time_to_first_token_seconds",
    "inter_token_latency_seconds",
    "e2e_request_latency_seconds",
    "request_queue_time_seconds",
)

_TYPE_RE = re.compile(r"^#\s*TYPE\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\w+)\s*$")
_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)" r"(?:\{(.*)\})?\s+" r"([^\s]+)" r"(?:\s+([+-]?\d+))?\s*$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class PrometheusSample:
    """One parsed sample from the Prometheus/OpenMetrics text format."""

    name: str
    value: float
    labels: Mapping[str, str] = field(default_factory=dict)
    timestamp_ms: Optional[int] = None
    metric_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "name": self.name,
            "labels": dict(self.labels),
            "value": self.value if math.isfinite(self.value) else None,
            "timestamp_ms": self.timestamp_ms,
            "metric_type": self.metric_type,
        }


@dataclass(frozen=True)
class PrometheusSnapshot:
    """A timestamped /metrics response, including explicit failure state."""

    monotonic_ns: int
    wall_time: str
    samples: tuple[PrometheusSample, ...] = ()
    error: Optional[str] = None
    raw_text: Optional[str] = None
    http_status: Optional[int] = None

    @property
    def endpoint_available(self) -> bool:
        """Whether the endpoint request and exposition parsing succeeded."""
        return self.error is None

    @property
    def metrics(self) -> dict[str, Any]:
        """Return stable, canonical vLLM metric keys for this snapshot."""
        return canonicalize_vllm_metrics(self.samples)

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        """Return a JSON-safe artifact record."""
        result: dict[str, Any] = {
            "monotonic_ns": self.monotonic_ns,
            "wall_time": self.wall_time,
            "available": self.endpoint_available,
            "error": self.error,
            "http_status": self.http_status,
            "metrics": self.metrics,
        }
        if include_raw:
            result["raw_text"] = self.raw_text
        return result


class PrometheusParser:
    """Parse Prometheus 0.0.4 and commonly emitted OpenMetrics samples."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def parse(self, text: str) -> list[PrometheusSample]:
        """Parse exposition text into individual labelled samples."""
        metric_types: dict[str, str] = {}
        parsed: list[PrometheusSample] = []

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line == "# EOF":
                continue
            type_match = _TYPE_RE.match(line)
            if type_match:
                metric_types[type_match.group(1)] = type_match.group(2).lower()
                continue
            if line.startswith("#"):
                continue

            # OpenMetrics exemplars follow the value after `` # {...}``.
            sample_part = line.split(" # ", 1)[0].strip()
            match = _SAMPLE_RE.match(sample_part)
            if not match:
                if self.strict:
                    raise ValueError(f"invalid Prometheus sample on line {line_number}: {line}")
                continue

            name, label_text, raw_value, raw_timestamp = match.groups()
            try:
                value = _parse_float(raw_value)
                labels = _parse_labels(label_text or "")
                timestamp_ms = int(raw_timestamp) if raw_timestamp is not None else None
            except ValueError:
                if self.strict:
                    raise ValueError(f"invalid Prometheus sample on line {line_number}: {line}")
                continue

            family = _metric_family(name, metric_types)
            parsed.append(
                PrometheusSample(
                    name=name,
                    value=value,
                    labels=labels,
                    timestamp_ms=timestamp_ms,
                    metric_type=metric_types.get(family),
                )
            )
        return parsed

    def parse_vllm(self, text: str) -> dict[str, Any]:
        """Parse text directly into the canonical vLLM metric mapping."""
        return canonicalize_vllm_metrics(self.parse(text))


# Descriptive alias for callers that only need the canonical vLLM view.
PrometheusMetricsParser = PrometheusParser


def _parse_float(value: str) -> float:
    lowered = value.lower()
    if lowered in ("+inf", "inf"):
        return math.inf
    if lowered == "-inf":
        return -math.inf
    if lowered in ("nan", "+nan", "-nan"):
        return math.nan
    return float(value)


def _unescape_label(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\" or index + 1 >= len(value):
            result.append(value[index])
            index += 1
            continue
        escaped = value[index + 1]
        result.append("\n" if escaped == "n" else escaped)
        index += 2
    return "".join(result)


def _parse_labels(text: str) -> dict[str, str]:
    if not text.strip():
        return {}
    labels: dict[str, str] = {}
    position = 0
    for match in _LABEL_RE.finditer(text):
        separator = text[position : match.start()].strip()
        if separator not in ("", ","):
            raise ValueError("invalid Prometheus labels")
        labels[match.group(1)] = _unescape_label(match.group(2))
        position = match.end()
    if text[position:].strip() not in ("", ",") or not labels:
        raise ValueError("invalid Prometheus labels")
    return labels


def _metric_family(name: str, metric_types: Mapping[str, str]) -> str:
    if name in metric_types:
        return name
    for suffix in ("_bucket", "_sum", "_count", "_created"):
        if name.endswith(suffix) and name[: -len(suffix)] in metric_types:
            return name[: -len(suffix)]
    return name


def parse_prometheus_text(text: str, strict: bool = False) -> list[PrometheusSample]:
    """Convenience function for parsing Prometheus exposition text."""
    return PrometheusParser(strict=strict).parse(text)


def parse_vllm_metrics(text: str, strict: bool = False) -> dict[str, Any]:
    """Parse exposition text and return stable vLLM metric aliases."""
    return canonicalize_vllm_metrics(parse_prometheus_text(text, strict=strict))


def _select_alias(
    samples: Sequence[PrometheusSample], aliases: Sequence[str], histogram: bool = False
) -> list[PrometheusSample]:
    suffixes = ("", "_bucket", "_sum", "_count", "_created") if histogram else ("",)
    for alias in aliases:
        names = {alias + suffix for suffix in suffixes}
        selected = [sample for sample in samples if sample.name in names]
        if selected:
            return selected
    return []


def _finite_sum(samples: Iterable[PrometheusSample]) -> Optional[float]:
    values = [sample.value for sample in samples if math.isfinite(sample.value)]
    return math.fsum(values) if values else None


def histogram_quantile(quantile: float, buckets: Mapping[float, float]) -> Optional[float]:
    """Estimate a histogram quantile from cumulative Prometheus buckets."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(
        (float(bound), float(count))
        for bound, count in buckets.items()
        if math.isfinite(float(count))
    )
    if not ordered or ordered[-1][1] <= 0:
        return None

    target = ordered[-1][1] * quantile
    previous_bound = 0.0
    previous_count = 0.0
    for bound, cumulative_count in ordered:
        if cumulative_count < target:
            if math.isfinite(bound):
                previous_bound = bound
            previous_count = cumulative_count
            continue
        if not math.isfinite(bound):
            return previous_bound if previous_count > 0 else None
        bucket_count = cumulative_count - previous_count
        if bucket_count <= 0:
            return bound
        fraction = (target - previous_count) / bucket_count
        return previous_bound + (bound - previous_bound) * fraction
    return None


def _histogram_value(samples: Sequence[PrometheusSample], alias: str) -> Optional[dict[str, Any]]:
    buckets: dict[float, float] = {}
    for sample in samples:
        if sample.name != alias + "_bucket" or not math.isfinite(sample.value):
            continue
        raw_bound = sample.labels.get("le")
        if raw_bound is None:
            continue
        try:
            bound = _parse_float(raw_bound)
        except ValueError:
            continue
        buckets[bound] = buckets.get(bound, 0.0) + sample.value

    count = _finite_sum(sample for sample in samples if sample.name == alias + "_count")
    total = _finite_sum(sample for sample in samples if sample.name == alias + "_sum")
    if count is None and math.inf in buckets:
        count = buckets[math.inf]

    # Prometheus summaries expose quantile-labelled base samples. Preserve them
    # when a vLLM build uses a summary instead of histogram buckets.
    quantiles: dict[float, float] = {}
    for sample in samples:
        if sample.name != alias or not math.isfinite(sample.value):
            continue
        raw_quantile = sample.labels.get("quantile")
        if raw_quantile is None:
            continue
        try:
            quantiles[float(raw_quantile)] = sample.value
        except ValueError:
            continue

    if count is None and total is None and not buckets and not quantiles:
        return None

    def quantile_value(level: float) -> Optional[float]:
        if level in quantiles:
            return quantiles[level]
        return histogram_quantile(level, buckets)

    mean = total / count if total is not None and count is not None and count != 0 else None
    return {
        "available": True,
        "count": count,
        "sum": total,
        "mean": mean,
        "buckets": {str(bound): value for bound, value in sorted(buckets.items())},
        "p50": quantile_value(0.5),
        "p95": quantile_value(0.95),
        "p99": quantile_value(0.99),
    }


def canonicalize_vllm_metrics(samples: Iterable[PrometheusSample]) -> dict[str, Any]:
    """Map version-specific vLLM names to stable keys.

    Missing metrics remain ``None``. If multiple labelled series exist, request
    counts/counters are summed while KV pressure uses the maximum series value.
    Aliases are priority ordered, preventing legacy and current names from being
    double-counted when both are exported.
    """
    sample_list = list(samples)
    result: dict[str, Any] = {name: None for name in VLLM_METRIC_ALIASES}

    for canonical_name in GAUGE_METRICS + COUNTER_METRICS:
        selected = _select_alias(sample_list, VLLM_METRIC_ALIASES[canonical_name])
        values = [sample.value for sample in selected if math.isfinite(sample.value)]
        if not values:
            continue
        result[canonical_name] = (
            max(values) if canonical_name == "kv_cache_usage_perc" else math.fsum(values)
        )

    for canonical_name in HISTOGRAM_METRICS:
        aliases = VLLM_METRIC_ALIASES[canonical_name]
        selected = _select_alias(sample_list, aliases, histogram=True)
        if not selected:
            continue
        selected_alias = next(
            alias
            for alias in aliases
            if any(
                sample.name == alias
                or sample.name in {alias + "_bucket", alias + "_sum", alias + "_count"}
                for sample in selected
            )
        )
        result[canonical_name] = _histogram_value(selected, selected_alias)
    return result


def _parse_bucket_mapping(histogram: Mapping[str, Any]) -> dict[float, float]:
    parsed: dict[float, float] = {}
    raw_buckets = histogram.get("buckets", {})
    if not isinstance(raw_buckets, Mapping):
        return parsed
    for raw_bound, raw_count in raw_buckets.items():
        try:
            bound = _parse_float(str(raw_bound))
            count = float(raw_count)
        except (TypeError, ValueError):
            continue
        if math.isfinite(count):
            parsed[bound] = count
    return parsed


def _histogram_window_delta(
    values: Sequence[Optional[Mapping[str, Any]]],
) -> dict[str, Any]:
    histograms = [value for value in values if value is not None]
    if len(histograms) < 2:
        return {
            "available": False,
            "sample_count": len(histograms),
            "count": None,
            "sum": None,
            "mean": None,
            "buckets": {},
            "p50": None,
            "p95": None,
            "p99": None,
            "reset_count": 0,
        }

    delta_count = 0.0
    delta_sum = 0.0
    count_available = True
    sum_available = True
    delta_buckets: dict[float, float] = {}
    reset_count = 0

    previous = histograms[0]
    for current in histograms[1:]:
        previous_count = previous.get("count")
        current_count = current.get("count")
        reset = False
        if previous_count is None or current_count is None:
            count_available = False
        else:
            previous_count_value = float(previous_count)
            current_count_value = float(current_count)
            reset = current_count_value < previous_count_value
            delta_count += (
                current_count_value if reset else current_count_value - previous_count_value
            )
        if reset:
            reset_count += 1

        previous_sum = previous.get("sum")
        current_sum = current.get("sum")
        if previous_sum is None or current_sum is None:
            sum_available = False
        else:
            previous_sum_value = float(previous_sum)
            current_sum_value = float(current_sum)
            delta_sum += current_sum_value if reset else current_sum_value - previous_sum_value

        previous_buckets = _parse_bucket_mapping(previous)
        current_buckets = _parse_bucket_mapping(current)
        for bound in previous_buckets.keys() | current_buckets.keys():
            if bound not in previous_buckets or bound not in current_buckets:
                continue
            contribution = (
                current_buckets[bound]
                if reset
                else current_buckets[bound] - previous_buckets[bound]
            )
            delta_buckets[bound] = delta_buckets.get(bound, 0.0) + max(contribution, 0.0)
        previous = current

    count_value = delta_count if count_available else None
    sum_value = delta_sum if sum_available else None
    mean = (
        sum_value / count_value
        if sum_value is not None and count_value is not None and count_value != 0
        else None
    )
    return {
        "available": count_value is not None or bool(delta_buckets),
        "sample_count": len(histograms),
        "count": count_value,
        "sum": sum_value,
        "mean": mean,
        "buckets": {str(bound): value for bound, value in sorted(delta_buckets.items())},
        "p50": histogram_quantile(0.5, delta_buckets),
        "p95": histogram_quantile(0.95, delta_buckets),
        "p99": histogram_quantile(0.99, delta_buckets),
        "reset_count": reset_count,
    }


def summarize_prometheus_snapshots(
    snapshots: Iterable[PrometheusSnapshot],
) -> dict[str, Any]:
    """Aggregate vLLM snapshots over exactly one measurement window."""
    snapshot_list = list(snapshots)
    canonical = [snapshot.metrics for snapshot in snapshot_list]
    gauges = {
        name: summarize_values(metrics.get(name) for metrics in canonical) for name in GAUGE_METRICS
    }
    counters = {
        name: counter_window_delta(metrics.get(name) for metrics in canonical)
        for name in COUNTER_METRICS
    }
    histograms = {
        name: _histogram_window_delta(
            [
                metrics.get(name) if isinstance(metrics.get(name), Mapping) else None
                for metrics in canonical
            ]
        )
        for name in HISTOGRAM_METRICS
    }
    direct: dict[str, Any] = {}
    direct.update(gauges)
    direct.update(counters)
    direct.update(histograms)
    missing = [name for name, summary in direct.items() if not summary["available"]]
    successful = sum(snapshot.endpoint_available for snapshot in snapshot_list)
    return {
        "available": successful > 0 and any(summary["available"] for summary in direct.values()),
        "sample_count": len(snapshot_list),
        "successful_sample_count": successful,
        "error_count": len(snapshot_list) - successful,
        "errors": [snapshot.error for snapshot in snapshot_list if snapshot.error],
        "missing_metrics": missing,
        "gauges": gauges,
        "counters": counters,
        "histograms": histograms,
        **direct,
    }


FetchMetrics = Callable[[], Union[str, Awaitable[str]]]


class PrometheusCollector:
    """Asynchronously fetch and parse a vLLM ``/metrics`` endpoint."""

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 2.0,
        client: Optional[httpx.AsyncClient] = None,
        fetcher: Optional[FetchMetrics] = None,
        parser: Optional[PrometheusParser] = None,
        include_raw: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = _normalise_metrics_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None
        self._fetcher = fetcher
        self.parser = parser or PrometheusParser()
        self.include_raw = include_raw
        self.raise_on_error = raise_on_error
        self.snapshots: list[PrometheusSnapshot] = []

    async def collect(self, timestamp: Optional[SampleTimestamp] = None) -> PrometheusSnapshot:
        """Collect one snapshot; failures become explicit unavailable samples."""
        status: Optional[int] = None
        raw_text: Optional[str] = None
        try:
            if self._fetcher is not None:
                fetched = self._fetcher()
                raw_text = await fetched if inspect.isawaitable(fetched) else fetched
            else:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False)
                response = await self._client.get(
                    self.endpoint,
                    headers={"Accept": "application/openmetrics-text, text/plain"},
                    timeout=self.timeout_seconds,
                )
                status = response.status_code
                response.raise_for_status()
                raw_text = response.text
            samples = tuple(self.parser.parse(raw_text))
            captured = timestamp or capture_timestamp()
            snapshot = PrometheusSnapshot(
                monotonic_ns=captured.monotonic_ns,
                wall_time=captured.wall_time,
                samples=samples,
                raw_text=raw_text if self.include_raw else None,
                http_status=status,
            )
        except Exception as error:
            if self.raise_on_error:
                raise
            captured = timestamp or capture_timestamp()
            snapshot = PrometheusSnapshot(
                monotonic_ns=captured.monotonic_ns,
                wall_time=captured.wall_time,
                error=f"{type(error).__name__}: {error}",
                raw_text=raw_text if self.include_raw else None,
                http_status=status,
            )
        self.snapshots.append(snapshot)
        return snapshot

    async def sample(self, timestamp: Optional[SampleTimestamp] = None) -> PrometheusSnapshot:
        """Alias for :meth:`collect`, convenient in sampling loops."""
        return await self.collect(timestamp=timestamp)

    def clear(self) -> None:
        """Clear snapshots from prior measurement windows."""
        self.snapshots.clear()

    async def close(self) -> None:
        """Close an internally-created HTTP client."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "PrometheusCollector":
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.close()


def _normalise_metrics_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("Prometheus endpoint must not be empty")
    if endpoint.rstrip("/").endswith("/metrics"):
        return endpoint
    return endpoint.rstrip("/") + "/metrics"


# Collector was the term used in the project plan; client is a familiar alias
# for integrations that treat this class as an HTTP client.
PrometheusClient = PrometheusCollector
