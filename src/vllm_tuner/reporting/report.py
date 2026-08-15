"""Evidence-first static SLOTune report generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

from .plots import (
    capacity_curve,
    figure_data_available,
    figure_unavailable_reason,
    save_figure,
    search_comparison,
    search_trajectory,
    summarize_capacity_rows,
    telemetry_timeline,
)


def _finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            result.append(number)
    return result


def comparison_rows(trials: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeats by exact candidate while preserving feasible counts."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for trial in trials:
        method = str(trial.get("method", "unknown"))
        repeat_of = trial.get("repeat_of")
        parameters = trial.get("parameters", {})
        candidate = (
            f"{method}-{repeat_of}:{json.dumps(parameters, sort_keys=True)}"
            if repeat_of is not None
            else method
        )
        grouped.setdefault(candidate, []).append(trial)
    rows: list[dict[str, Any]] = []
    for candidate, values in sorted(grouped.items()):
        feasible = [
            value
            for value in values
            if value.get("status") == "COMPLETE" and value.get("feasible", True)
        ]
        goodputs = _finite(value.get("goodput_requests_per_sec") for value in feasible)
        p99_ttft = _finite(value.get("p99_ttft_ms") for value in feasible)
        rows.append(
            {
                "method": candidate.split(":", 1)[0],
                "trials": len(values),
                "feasible_trials": len(feasible),
                "median_goodput": median(goodputs) if goodputs else None,
                "min_goodput": min(goodputs) if goodputs else None,
                "max_goodput": max(goodputs) if goodputs else None,
                "median_p99_ttft_ms": median(p99_ttft) if p99_ttft else None,
            }
        )
    return rows


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _aggregate_cell(row: Mapping[str, Any], phase: str, metric: str) -> str:
    aggregates = row.get(f"{phase}_metrics")
    if not isinstance(aggregates, Mapping):
        return "unavailable"
    record = aggregates.get(metric)
    if not isinstance(record, Mapping):
        return "unavailable"
    return (
        f"{_display(record.get('median'))} "
        f"({_display(record.get('min'))}–{_display(record.get('max'))}; "
        f"n={_display(record.get('count'))})"
    )


def markdown_comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the default/random/TPE comparison table."""
    lines = [
        "| Method | Trials | Feasible | Median goodput (req/s) | Range | Median p99 TTFT (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        range_text = (
            f"{_display(row.get('min_goodput'))}–{_display(row.get('max_goodput'))}"
            if row.get("min_goodput") is not None
            else "unavailable"
        )
        lines.append(
            "| {method} | {trials} | {feasible} | {median_goodput} | {range_text} | "
            "{p99} |".format(
                method=row["method"],
                trials=row["trials"],
                feasible=row["feasible_trials"],
                median_goodput=_display(row.get("median_goodput")),
                range_text=range_text,
                p99=_display(row.get("median_p99_ttft_ms")),
            )
        )
    return "\n".join(lines) + "\n"


def _scheduler_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Trace | Policy | Budget | Goodput | p99 TTFT | p99 TPOT | Fairness | Starvation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {trace} | {policy} | {budget} | {goodput} | {ttft} | {tpot} | "
            "{fairness} | {starvation} |".format(
                trace=row.get("trace", "unknown"),
                policy=row.get("policy", "unknown"),
                budget=_display(row.get("budget")),
                goodput=_display(row.get("goodput")),
                ttft=_display(row.get("p99_ttft")),
                tpot=_display(row.get("p99_tpot")),
                fairness=_display(row.get("fairness_index")),
                starvation=_display(row.get("starvation_count")),
            )
        )
    return "\n".join(lines) + "\n" if rows else "Data unavailable.\n"


def _trial_records_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "Data unavailable.\n"
    lines = [
        "| Trial | Method | Status | Feasible | Goodput | p99 TTFT | p99 TPOT | p99 E2E |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {method} | {status} | {feasible} | {goodput} | {ttft} | "
            "{tpot} | {e2e} |".format(
                trial=row.get("trial_id", row.get("trial_number", "unknown")),
                method=row.get("method", "unknown"),
                status=row.get("status", "unknown"),
                feasible=_display(row.get("feasible")),
                goodput=_display(row.get("goodput_requests_per_sec")),
                ttft=_display(row.get("p99_ttft_ms")),
                tpot=_display(row.get("p99_tpot_ms")),
                e2e=_display(row.get("p99_e2e_ms")),
            )
        )
    return "\n".join(lines) + "\n"


def _validation_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "Data unavailable; no candidate validation records were produced.\n"
    lines = [
        "| Candidate | Repeats feasible | Repeat goodput median (range) | Repeat p99 TTFT median (range) | Holdouts feasible | Holdout goodput median (range) | Holdout p99 TTFT median (range) | Ratio | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {repeat_count}/{repeat_required} | {repeat_goodput} | "
            "{repeat_ttft} | {holdout_count}/{holdout_required} | {holdout_goodput} | "
            "{holdout_ttft} | {ratio} | {gate} |".format(
                candidate=row.get("candidate", "unknown"),
                repeat_count=row.get("repeat_complete_feasible", 0),
                repeat_required=row.get("repeat_required", 0),
                repeat_goodput=_aggregate_cell(row, "repeat", "goodput_requests_per_sec"),
                repeat_ttft=_aggregate_cell(row, "repeat", "p99_ttft_ms"),
                holdout_count=row.get("holdout_complete_feasible", 0),
                holdout_required=(
                    row.get("repeat_required", 0) if row.get("holdout_required") else 0
                ),
                holdout_goodput=_aggregate_cell(row, "holdout", "goodput_requests_per_sec"),
                holdout_ttft=_aggregate_cell(row, "holdout", "p99_ttft_ms"),
                ratio=_display(row.get("holdout_to_repeat_goodput_ratio")),
                gate="PASS" if row.get("validated") is True else "FAIL",
            )
        )
    return "\n".join(lines) + "\n"


def _capacity_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    summaries = summarize_capacity_rows(rows)
    if not summaries:
        return "Data unavailable; no explicit default-config capacity rates were run.\n"
    lines = [
        "| Offered req/s | Repeats | Complete | Feasible | Failed | Median achieved (range) | Median goodput (range) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        achieved = (
            f"{_display(row['median_achieved_requests_per_sec'])} "
            f"({_display(row['min_achieved_requests_per_sec'])}–"
            f"{_display(row['max_achieved_requests_per_sec'])})"
            if row["median_achieved_requests_per_sec"] is not None
            else "unavailable"
        )
        goodput = (
            f"{_display(row['median_goodput_requests_per_sec'])} "
            f"({_display(row['min_goodput_requests_per_sec'])}–"
            f"{_display(row['max_goodput_requests_per_sec'])})"
            if row["median_goodput_requests_per_sec"] is not None
            else "unavailable"
        )
        lines.append(
            f"| {_display(row['offered_requests_per_sec'])} | {row['repeat_count']} | "
            f"{row['complete_count']} | {row['feasible_count']} | {row['failed_count']} | "
            f"{achieved} | {goodput} |"
        )
    return "\n".join(lines) + "\n"


def _available_series_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        row.get("record_type") != "availability" and row.get("available") is not False
        for row in rows
    )


def _generate_plots(
    destination: Path,
    capacity_rows: Sequence[Mapping[str, Any]],
    trial_rows: Sequence[Mapping[str, Any]],
    client_rows: Sequence[Mapping[str, Any]],
    engine_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Path], dict[str, Any]]:
    figures = {
        "capacity_curve": (
            "capacity-curve",
            capacity_curve(capacity_rows),
        ),
        "pareto": ("pareto", search_comparison(trial_rows)),
        "search_trajectory": (
            "search-trajectory",
            search_trajectory(trial_rows),
        ),
        "telemetry_timeline": (
            "telemetry-timeline",
            telemetry_timeline(client_rows, engine_rows, gpu_rows),
        ),
    }
    paths: dict[str, Path] = {}
    plot_manifest: dict[str, Any] = {"schema_version": 1, "plots": {}}
    for key, (stem, figure) in figures.items():
        html_path = destination / f"{stem}.html"
        png_path = destination / f"{stem}.png"
        outputs = save_figure(figure, html_path, png_path)
        paths[f"{key}_html"] = html_path
        if "png" in outputs:
            paths[f"{key}_png"] = png_path
        plot_manifest["plots"][key] = {
            "data_available": figure_data_available(figure),
            "unavailable_reason": figure_unavailable_reason(figure),
            "html": html_path.name,
            "png": png_path.name if "png" in outputs else None,
            "static_image_available": outputs.get("static_image_available") == "true",
            "fallback_reason": outputs.get("fallback_reason"),
            "data_source": (
                "aggregate/capacity-sweep.parquet"
                if key == "capacity_curve"
                else "search trials" if key != "telemetry_timeline" else "trial raw JSONL"
            ),
        }
    manifest_path = destination / "plot-manifest.json"
    manifest_path.write_text(
        json.dumps(plot_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths["plot_manifest"] = manifest_path
    return paths, plot_manifest


def _figures_markdown(plot_manifest: Mapping[str, Any]) -> str:
    labels = {
        "capacity_curve": "Capacity curve",
        "pareto": "TTFT–goodput comparison",
        "search_trajectory": "Search trajectory",
        "telemetry_timeline": "Cross-layer telemetry timeline",
    }
    lines: list[str] = []
    plots = plot_manifest.get("plots", {})
    for key, label in labels.items():
        record = plots.get(key, {}) if isinstance(plots, Mapping) else {}
        html_name = record.get("html")
        png_name = record.get("png")
        if not record.get("data_available", False):
            lines.append(
                f"- {label}: unavailable ({record.get('unavailable_reason') or 'no reason recorded'}); "
                f"[diagnostic HTML]({html_name})."
            )
        elif png_name:
            lines.append(f"- {label}: [interactive HTML]({html_name})")
            lines.append(f"\n  ![{label}]({png_name})")
        else:
            lines.append(
                f"- {label}: [interactive HTML fallback]({html_name}); PNG unavailable "
                f"({record.get('fallback_reason') or 'renderer unavailable'})."
            )
    return "\n".join(lines)


def generate_report(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    trials: Iterable[Mapping[str, Any]],
    repetitions: Iterable[Mapping[str, Any]] = (),
    holdout: Iterable[Mapping[str, Any]] = (),
    candidate_validation: Iterable[Mapping[str, Any]] = (),
    scheduler_results: Iterable[Mapping[str, Any]] = (),
    capacity_sweep: Iterable[Mapping[str, Any]] = (),
    limitations: Optional[list[str]] = None,
    client_series: Iterable[Mapping[str, Any]] = (),
    engine_series: Iterable[Mapping[str, Any]] = (),
    gpu_series: Iterable[Mapping[str, Any]] = (),
    telemetry_source: Optional[str] = None,
) -> dict[str, Path]:
    """Generate self-contained Markdown and HTML without inventing unavailable results."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trial_rows = list(trials)
    repeat_rows = list(repetitions)
    holdout_rows = list(holdout)
    validation_rows = list(candidate_validation)
    scheduling_rows = list(scheduler_results)
    capacity_rows = list(capacity_sweep)
    client_rows = list(client_series)
    engine_rows = list(engine_series)
    gpu_rows = list(gpu_series)
    table = comparison_rows(repeat_rows or trial_rows)
    comparison = markdown_comparison(table)
    (destination / "comparison-table.md").write_text(comparison, encoding="utf-8")

    limitations = list(limitations or [])
    if not holdout_rows:
        limitations.append("No holdout results were supplied; no generalization claim is made.")
    if not repeat_rows:
        limitations.append(
            "No repeated candidate results were supplied; variability is unavailable."
        )
    plot_paths, plot_manifest = _generate_plots(
        destination,
        capacity_rows,
        trial_rows,
        client_rows,
        engine_rows,
        gpu_rows,
    )
    telemetry_record = plot_manifest["plots"]["telemetry_timeline"]
    if not telemetry_record["data_available"]:
        limitations.append(
            "Raw telemetry for the selected trial was unavailable or not plottable; the timeline is explicitly marked unavailable."
        )
    figure_markdown = _figures_markdown(plot_manifest)
    scheduler_markdown = _scheduler_markdown(scheduling_rows)
    holdout_markdown = _trial_records_markdown(holdout_rows)
    validation_markdown = _validation_markdown(validation_rows)
    capacity_markdown = _capacity_markdown(capacity_rows)

    markdown = f"""# SLOTune experiment report

## Experiment identity

- Experiment: `{manifest.get('experiment_id', 'unknown')}`
- Model: `{manifest.get('model', 'unknown')}`
- Source commit: `{manifest.get('source_commit') or 'unavailable'}`
- Trace SHA-256: `{manifest.get('trace_sha256', 'unavailable')}`
- Holdout trace SHA-256: `{manifest.get('holdout_trace_sha256') or 'unavailable'}`
- Search-space SHA-256: `{manifest.get('search_space_sha256', 'unavailable')}`

## Methodology

SLOTune maximizes successful requests that individually satisfy the configured TTFT, TPOT, and
E2E SLOs per measurement second. Request errors, OOM, server exit, p99 SLO violations, and VRAM
limits are hard constraints. Offered load, achieved throughput, and SLO goodput are reported as
different quantities. Warmup requests are excluded and raw request/engine/GPU series remain in
the trial artifact directory.

## Default vs random vs constrained TPE

{comparison}

## Figures

{figure_markdown}

## Default capacity sweep

Only trials using vLLM default tunable parameters are included. Values are medians and full ranges
across repeats at the configured offered request rate; search candidates are never mixed in.

{capacity_markdown}

## Holdout validation

Holdout records: {len(holdout_rows)}. Re-running alone is not validation. A candidate passes only
when every configured repeat and every exact-parameter holdout run is COMPLETE and feasible, and
the holdout/repeat median goodput ratio meets the recorded threshold.

{validation_markdown}

{holdout_markdown}

## Adaptive token-budget experiment

Scheduler records: {len(scheduling_rows)}. Fixed-budget and adaptive rows include queue tails,
goodput, fairness, starvation, and preemption; negative deltas are retained.

{scheduler_markdown}

## Telemetry evidence

- Source trial: `{telemetry_source or 'unavailable'}`
- Client measured rows: {_available_series_count(client_rows)}
- Engine measured rows: {_available_series_count(engine_rows)}
- GPU measured rows: {_available_series_count(gpu_rows)}

## Limitations

"""
    markdown += "\n".join(f"- {item}" for item in limitations) or "- None recorded."
    markdown += "\n\n## Upstream and contribution boundary\n\n"
    markdown += (
        "Forked from jranaraki/vllm-tuner. This work adds benchmark correctness, SLO-aware "
        "optimization, cross-layer observability, reproducibility, and scheduling experiments.\n"
    )
    markdown_path = destination / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    escaped = html.escape(markdown)
    payload = html.escape(
        json.dumps(
            {
                "manifest": dict(manifest),
                "trials": trial_rows,
                "repetitions": repeat_rows,
                "holdout": holdout_rows,
                "candidate_validation": validation_rows,
                "scheduler": scheduling_rows,
                "capacity_sweep": capacity_rows,
            },
            ensure_ascii=False,
        )
    )
    html_path = destination / "report.html"
    plot_links = "".join(
        f"<li><a href='{html.escape(str(record.get('html')))}'>{html.escape(str(key))}</a>"
        + (
            f" — <a href='{html.escape(str(record.get('png')))}'>PNG</a>"
            if record.get("png")
            else " — HTML fallback"
        )
        + "</li>"
        for key, record in plot_manifest["plots"].items()
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>SLOTune report</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5;}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem;}details{margin-top:2rem;}</style>"
        f"</head><body><h1>SLOTune experiment report</h1><h2>Figures</h2><ul>{plot_links}</ul>"
        f"<pre>{escaped}</pre><details><summary>Machine-readable evidence</summary>"
        f"<pre>{payload}</pre></details></body></html>",
        encoding="utf-8",
    )
    return {
        "markdown": markdown_path,
        "html": html_path,
        "comparison": destination / "comparison-table.md",
        **plot_paths,
    }
