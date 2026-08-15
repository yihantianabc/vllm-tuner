"""Capacity, search, and cross-layer telemetry figures with honest missing data."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested(row: Mapping[str, Any], paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = row
        for component in path.split("."):
            if not isinstance(value, Mapping) or component not in value:
                value = None
                break
            value = value[component]
        if value is not None:
            return value
    return None


def _set_availability(figure: go.Figure, available: bool, reason: str) -> None:
    figure.update_layout(
        meta={
            "data_available": available,
            "unavailable_reason": None if available else reason,
        }
    )
    if not available:
        figure.add_annotation(
            text=f"Data unavailable: {reason}",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#8a3b12", "size": 15},
            bgcolor="#fff3e8",
            bordercolor="#d8a47f",
            borderwidth=1,
        )


def figure_data_available(figure: go.Figure) -> bool:
    """Return the availability flag attached by SLOTune plotting helpers."""
    metadata = figure.layout.meta
    return bool(metadata.get("data_available", False)) if isinstance(metadata, dict) else False


def figure_unavailable_reason(figure: go.Figure) -> Optional[str]:
    """Return a human-readable reason when a figure has no measured points."""
    metadata = figure.layout.meta
    if not isinstance(metadata, dict):
        return "figure did not expose availability metadata"
    value = metadata.get("unavailable_reason")
    return str(value) if value else None


def capacity_curve(rows: Iterable[Mapping[str, Any]]) -> go.Figure:
    """Plot default-config capacity medians/ranges across explicit rate repeats."""
    data = summarize_capacity_rows(rows)

    figure = go.Figure()
    measured_points = 0
    for field, label in (
        ("achieved_requests_per_sec", "Achieved throughput"),
        ("goodput_requests_per_sec", "SLO goodput"),
    ):
        median_field = f"median_{field}"
        min_field = f"min_{field}"
        max_field = f"max_{field}"
        points = [row for row in data if _number(row.get(median_field)) is not None]
        if not points:
            continue
        measured_points += len(points)
        medians = [_number(row.get(median_field)) for row in points]
        assert all(value is not None for value in medians)
        figure.add_trace(
            go.Scatter(
                x=[row["offered_requests_per_sec"] for row in points],
                y=medians,
                mode="lines+markers",
                name=f"{label} median",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [float(row[max_field]) - float(row[median_field]) for row in points],
                    "arrayminus": [
                        float(row[median_field]) - float(row[min_field]) for row in points
                    ],
                    "visible": True,
                },
                customdata=[row["repeat_count"] for row in points],
                hovertemplate=(
                    "Offered=%{x:.3f}<br>Median=%{y:.3f}<br>Repeats=%{customdata}" "<extra></extra>"
                ),
            )
        )

    offered_values = [row["offered_requests_per_sec"] for row in data]
    if offered_values:
        figure.add_trace(
            go.Scatter(
                x=offered_values,
                y=offered_values,
                mode="lines",
                line={"dash": "dot", "color": "#777"},
                name="Offered load",
            )
        )
    figure.update_layout(
        title="Default vLLM capacity sweep (median and range across repeats)",
        xaxis_title="Offered requests/s",
        yaxis_title="Requests/s",
        template="plotly_white",
    )
    _set_availability(
        figure,
        measured_points > 0,
        "no explicit default-config capacity sweep measurements were recorded",
    )
    return figure


def summarize_capacity_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate capacity repeats by offered rate while retaining failures."""
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        offered = _number(row.get("offered_requests_per_sec"))
        if offered is not None:
            grouped.setdefault(offered, []).append(row)

    summaries: list[dict[str, Any]] = []
    for offered, values in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "offered_requests_per_sec": offered,
            "repeat_count": len(values),
            "complete_count": sum(row.get("status") == "COMPLETE" for row in values),
            "feasible_count": sum(bool(row.get("feasible", False)) for row in values),
            "failed_count": sum(row.get("status") == "FAILED" for row in values),
        }
        measured_values = [row for row in values if row.get("status") in {"COMPLETE", "INFEASIBLE"}]
        summary["measured_count"] = len(measured_values)
        for field in (
            "achieved_requests_per_sec",
            "goodput_requests_per_sec",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "p50_ttft_ms",
            "p95_ttft_ms",
            "p99_ttft_ms",
            "p50_tpot_ms",
            "p95_tpot_ms",
            "p99_tpot_ms",
            "p50_e2e_ms",
            "p95_e2e_ms",
            "p99_e2e_ms",
            "completed_requests",
            "failed_requests",
            "error_rate",
            "timeout_count",
            "peak_waiting_requests",
            "peak_kv_cache_usage",
            "preemptions",
            "peak_memory_mb",
            "p95_memory_mb",
            "mean_gpu_utilization_percent",
            "energy_joules",
            "energy_per_output_token_joules",
        ):
            measured = [
                number for row in measured_values if (number := _number(row.get(field))) is not None
            ]
            summary[f"median_{field}"] = median(measured) if measured else None
            summary[f"min_{field}"] = min(measured) if measured else None
            summary[f"max_{field}"] = max(measured) if measured else None
        summaries.append(summary)
    return summaries


def search_comparison(rows: Iterable[Mapping[str, Any]]) -> go.Figure:
    """Compare measured p99 TTFT and goodput; omit unavailable candidates."""
    figure = go.Figure()
    point_count = 0
    for row in rows:
        ttft = _number(row.get("p99_ttft_ms"))
        goodput = _number(row.get("goodput_requests_per_sec"))
        if ttft is None or goodput is None:
            continue
        point_count += 1
        figure.add_trace(
            go.Scatter(
                x=[ttft],
                y=[goodput],
                mode="markers",
                marker={
                    "size": 10,
                    "symbol": "circle" if row.get("feasible", False) else "x",
                },
                name=str(row.get("label", row.get("method", "trial"))),
                text=[str(row.get("parameters", {}))],
            )
        )
    figure.update_layout(
        title="TTFT–goodput comparison",
        xaxis_title="p99 TTFT (ms, lower is better)",
        yaxis_title="SLO goodput (requests/s, higher is better)",
        template="plotly_white",
    )
    _set_availability(
        figure,
        point_count > 0,
        "no trial contained both measured p99 TTFT and SLO goodput",
    )
    return figure


def search_trajectory(rows: Iterable[Mapping[str, Any]]) -> go.Figure:
    """Plot the measured search objective across trial order for each method."""
    data = list(rows)
    objective_field = (
        "goodput_requests_per_sec"
        if any(_number(row.get("goodput_requests_per_sec")) is not None for row in data)
        else "objective"
    )
    methods = sorted({str(row.get("method", "unknown")) for row in data})
    figure = go.Figure()
    point_count = 0
    for method in methods:
        points: list[tuple[float, float, str]] = []
        for position, row in enumerate(data):
            if str(row.get("method", "unknown")) != method:
                continue
            trial_number = _number(row.get("trial_number"))
            objective = _number(row.get(objective_field))
            if objective is None:
                continue
            points.append(
                (
                    trial_number if trial_number is not None else float(position),
                    objective,
                    str(row.get("status", "unknown")),
                )
            )
        points.sort(key=lambda point: point[0])
        if not points:
            continue
        point_count += len(points)
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                text=[point[2] for point in points],
                mode="lines+markers",
                name=method,
            )
        )
    figure.update_layout(
        title="Search trajectory",
        xaxis_title="Trial number",
        yaxis_title=(
            "SLO goodput (requests/s)"
            if objective_field == "goodput_requests_per_sec"
            else "Search objective"
        ),
        template="plotly_white",
    )
    _set_availability(
        figure,
        point_count > 0,
        "no finite goodput or objective values were recorded",
    )
    return figure


def telemetry_timeline(
    client: Iterable[Mapping[str, Any]],
    engine: Iterable[Mapping[str, Any]],
    gpu: Iterable[Mapping[str, Any]],
) -> go.Figure:
    """Align client latency, engine queue/KV, and GPU series on perf-counter time."""
    client_rows = list(client)
    engine_rows = list(engine)
    gpu_rows = list(gpu)
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
        subplot_titles=(
            "Client latency",
            "Engine queue and KV",
            "GPU utilization and VRAM",
        ),
    )

    definitions = (
        (
            client_rows,
            ("ttft_ms",),
            ("first_token_at", "sent_at", "monotonic_ns"),
            "client.ttft_ms",
            1,
            False,
            "markers",
        ),
        (
            engine_rows,
            ("metrics.num_requests_waiting", "num_requests_waiting"),
            ("monotonic_ns",),
            "engine.waiting",
            2,
            False,
            "lines",
        ),
        (
            engine_rows,
            (
                "metrics.kv_cache_usage_perc",
                "kv_cache_usage_perc",
                "kv_cache_usage",
            ),
            ("monotonic_ns",),
            "engine.kv_cache_usage",
            2,
            True,
            "lines",
        ),
        (
            gpu_rows,
            ("gpu_utilization_percent", "gpu_utilization"),
            ("monotonic_ns",),
            "gpu.utilization_percent",
            3,
            False,
            "lines",
        ),
        (
            gpu_rows,
            ("memory_used_mb",),
            ("monotonic_ns",),
            "gpu.memory_used_mb",
            3,
            True,
            "lines",
        ),
    )
    materialized: list[tuple[list[tuple[float, float]], str, int, bool, str]] = []
    timestamps: list[float] = []
    for rows, value_paths, timestamp_paths, label, subplot, secondary, mode in definitions:
        points: list[tuple[float, float]] = []
        for row in rows:
            timestamp = _number(_nested(row, timestamp_paths))
            value = _number(_nested(row, value_paths))
            if timestamp is None or value is None:
                continue
            points.append((timestamp, value))
            timestamps.append(timestamp)
        materialized.append((points, label, subplot, secondary, mode))

    origin = min(timestamps) if timestamps else 0.0
    point_count = 0
    for points, label, subplot, secondary, mode in materialized:
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        point_count += len(points)
        figure.add_trace(
            go.Scatter(
                x=[(point[0] - origin) / 1_000_000_000.0 for point in points],
                y=[point[1] for point in points],
                mode=mode,
                name=label,
            ),
            row=subplot,
            col=1,
            secondary_y=secondary,
        )

    figure.update_yaxes(title_text="TTFT (ms)", row=1, col=1)
    figure.update_yaxes(title_text="Waiting requests", row=2, col=1, secondary_y=False)
    figure.update_yaxes(title_text="KV usage", row=2, col=1, secondary_y=True)
    figure.update_yaxes(title_text="GPU util (%)", row=3, col=1, secondary_y=False)
    figure.update_yaxes(title_text="VRAM (MB)", row=3, col=1, secondary_y=True)
    figure.update_xaxes(title_text="Seconds from first recorded sample", row=3, col=1)
    figure.update_layout(
        title="Cross-layer telemetry timeline",
        template="plotly_white",
        height=900,
    )
    _set_availability(
        figure,
        point_count > 0,
        "client, engine, and GPU raw series did not contain plottable measurements",
    )
    return figure


def save_figure(
    figure: go.Figure,
    html_path: str | Path,
    png_path: Optional[str | Path] = None,
) -> dict[str, str]:
    """Save self-contained HTML and use it as a reliable PNG fallback."""
    html = Path(html_path)
    html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(html), include_plotlyjs=True, full_html=True)
    outputs = {"html": str(html)}
    if png_path is None:
        return outputs

    png = Path(png_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.write_image(str(png), width=1200, height=700, scale=2)
    except Exception as error:
        # Kaleido/Chrome is optional. Preserve the interactive, self-contained
        # artifact and record why a static image could not be produced.
        with contextlib.suppress(OSError):
            if png.exists():
                png.unlink()
        outputs["static_image_available"] = "false"
        message = " ".join(str(error).split())
        outputs["fallback_reason"] = f"{type(error).__name__}: {message[:500]}"
    else:
        outputs["png"] = str(png)
        outputs["static_image_available"] = "true"
    return outputs
