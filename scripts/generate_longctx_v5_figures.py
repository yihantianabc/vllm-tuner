#!/usr/bin/env python3
"""Generate the checked-in long-context v5 figures from sealed summaries."""

from __future__ import annotations

import argparse
import json
import statistics
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_ARTIFACT_ROOT = Path("/root/autodl-tmp/longctx-v5-artifacts")
DEFAULT_PLANNER = DEFAULT_ARTIFACT_ROOT / "longctx-v5-m1-planner-init-002"
DEFAULT_APC = DEFAULT_ARTIFACT_ROOT / "longctx-v5-m3-apc-formal-001"
DEFAULT_M5 = DEFAULT_ARTIFACT_ROOT / "longctx-v5-m5-decode-tail-engineering-001"
COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "slate": "#64748b",
    "light": "#e2e8f0",
}


def _load_summary(root: Path) -> dict[str, Any]:
    path = root / "summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"summary root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _metric(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _style(figure: go.Figure, title: str, height: int = 540) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 22}},
        template="plotly_white",
        width=1200,
        height=height,
        margin={"l": 80, "r": 45, "t": 90, "b": 75},
        font={"family": "DejaVu Sans, Arial, sans-serif", "size": 14, "color": "#0f172a"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "x": 0.5, "xanchor": "center"},
        hovermode=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["light"], zeroline=False)


def _planner_figure(summary: Mapping[str, Any]) -> go.Figure:
    validations = _sequence(summary.get("validations"), "planner validations")
    label_by_run = {
        "heldout-util-90-32k-r0": "Held-out 32K",
        "extrapolate-context-8k-r0": "Extrapolated 8K",
        "extrapolate-context-16k-r0": "Extrapolated 16K",
    }
    rows = []
    for value in validations:
        row = _mapping(value, "planner validation")
        run_id = str(row.get("run_id"))
        if run_id not in label_by_run:
            continue
        rows.append((label_by_run[run_id], row))
    if len(rows) != 3:
        raise ValueError("expected the held-out 32K and two context extrapolation validations")

    plan = _mapping(summary.get("deployment_plan"), "deployment plan")
    contexts = _sequence(plan.get("contexts"), "deployment contexts")
    context_labels = []
    safe_concurrency = []
    for value in contexts:
        context = _mapping(value, "deployment context")
        name = str(context.get("name", "unknown"))
        context_labels.append(
            name.replace("short-", "").replace("medium-", "").replace("long-", "")
        )
        safe_concurrency.append(
            _metric(context.get("safe_integer_concurrency"), "safe concurrency")
        )

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Prediction error on unseen profiles", "Safe memory-bound concurrency"),
        horizontal_spacing=0.16,
    )
    labels = [label for label, _ in rows]
    block_errors = [abs(_metric(row.get("block_error_percent"), "block error")) for _, row in rows]
    concurrency_errors = [
        abs(_metric(row.get("max_concurrency_error_percent"), "concurrency error"))
        for _, row in rows
    ]
    figure.add_trace(
        go.Bar(
            name="KV block error",
            x=labels,
            y=block_errors,
            marker_color=COLORS["blue"],
            text=[f"{value:.4f}%" for value in block_errors],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            name="Max-concurrency error",
            x=labels,
            y=concurrency_errors,
            marker_color=COLORS["cyan"],
            text=[f"{value:.4f}%" for value in concurrency_errors],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            name="Safe sequences",
            x=context_labels,
            y=safe_concurrency,
            marker_color=COLORS["green"],
            text=[str(int(value)) for value in safe_concurrency],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="Absolute error (%)", range=[0, 0.052], row=1, col=1)
    figure.update_yaxes(title_text="Concurrent full contexts", range=[0, 26], row=1, col=2)
    _style(
        figure,
        "KV Capacity Planner — ≤0.0446% observed error vs 10% target",
    )
    figure.add_annotation(
        text="Safe usable capacity: 192,880 tokens (BF16 KV, block size 16)",
        x=0.77,
        y=-0.19,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": COLORS["slate"]},
    )
    return figure


def _apc_figure(summary: Mapping[str, Any]) -> go.Figure:
    analysis = _mapping(summary.get("analysis"), "APC analysis")
    paired = _sequence(analysis.get("paired"), "APC paired results")
    by_prefix: dict[int, list[Mapping[str, Any]]] = {2048: [], 4096: []}
    for value in paired:
        row = _mapping(value, "APC paired row")
        prefix = int(_metric(row.get("prefix_tokens"), "prefix tokens"))
        if prefix in by_prefix:
            by_prefix[prefix].append(row)
    if any(len(rows) != 3 for rows in by_prefix.values()):
        raise ValueError("expected APC rows for 0%, 50%, and 100% reuse at 2K and 4K")

    figure = go.Figure()
    for prefix, color in ((2048, COLORS["blue"]), (4096, COLORS["green"])):
        rows = sorted(by_prefix[prefix], key=lambda row: float(row["reuse_percent"]))
        medians = []
        error_plus = []
        error_minus = []
        reuse = []
        for row in rows:
            changes = sorted(
                -_metric(_mapping(pair, "APC pair").get("warm_ttft_change_percent"), "TTFT change")
                for pair in _sequence(row.get("pairs"), "APC pairs")
            )
            median = float(statistics.median(changes))
            medians.append(median)
            error_plus.append(max(changes) - median)
            error_minus.append(median - min(changes))
            reuse.append(f"{int(float(row['reuse_percent']))}%")
        figure.add_trace(
            go.Bar(
                name=f"{prefix // 1024}K prefix",
                x=reuse,
                y=medians,
                marker_color=color,
                text=[f"{value:+.1f}%" for value in medians],
                textposition="auto",
                cliponaxis=False,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": error_plus,
                    "arrayminus": error_minus,
                    "color": COLORS["slate"],
                    "thickness": 1.2,
                },
            )
        )
    figure.add_hline(y=0, line_color=COLORS["slate"], line_width=1)
    figure.update_yaxes(title_text="Warm TTFT improvement vs APC off (%)", range=[-5, 62])
    figure.update_xaxes(title_text="Shared-prefix reuse")
    _style(figure, "Automatic Prefix Caching — benefit scales with real prefix reuse")
    figure.add_annotation(
        text="Bars: paired median; whiskers: three-repeat range. Goodput was not lower in 18/18 pairs.",
        x=0.5,
        y=-0.19,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": COLORS["slate"]},
    )
    return figure


def _m5_figure(summary: Mapping[str, Any]) -> go.Figure:
    analysis = _mapping(summary.get("analysis"), "M5 analysis")
    groups = _sequence(analysis.get("groups"), "M5 groups")
    paired = _sequence(analysis.get("paired"), "M5 paired results")
    group_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in groups:
        row = _mapping(value, "M5 group")
        group_by_key[(str(row.get("cohort_id")), str(row.get("profile_id")))] = row

    cohorts = ["target", "held-out"]
    cohort_labels = ["Target", "Held-out"]
    baseline_itl = []
    candidate_itl = []
    metrics: dict[str, list[float]] = {
        "ITL p99 improvement": [],
        "Long TTFT cost": [],
        "Decode TPOT cost": [],
        "Goodput loss": [],
    }
    paired_by_cohort = {
        str(_mapping(value, "M5 paired row").get("cohort_id")): _mapping(value, "M5 paired row")
        for value in paired
    }
    for cohort in cohorts:
        baseline = _mapping(
            group_by_key[(cohort, "production-default")].get("decode_interference_itl_p99_ms"),
            "baseline ITL",
        )
        candidate = _mapping(
            group_by_key[(cohort, "decode-tail-1024")].get("decode_interference_itl_p99_ms"),
            "candidate ITL",
        )
        baseline_itl.append(_metric(baseline.get("median"), "baseline ITL median"))
        candidate_itl.append(_metric(candidate.get("median"), "candidate ITL median"))
        aggregates = _mapping(paired_by_cohort[cohort].get("aggregates"), "M5 aggregates")
        keys = {
            "ITL p99 improvement": "decode_interference_itl_p99_improvement_percent",
            "Long TTFT cost": "long_prefill_ttft_p99_degradation_percent",
            "Decode TPOT cost": "decode_tpot_p99_degradation_percent",
            "Goodput loss": "decode_goodput_change_percent",
        }
        for label, key in keys.items():
            metric = _mapping(aggregates.get(key), key)
            value = _metric(metric.get("median"), f"{key} median")
            metrics[label].append(-value if label == "Goodput loss" else value)

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Interference-window Decode ITL p99", "Paired median benefit and costs"),
        horizontal_spacing=0.15,
    )
    figure.add_trace(
        go.Bar(
            name="Production default",
            x=cohort_labels,
            y=baseline_itl,
            marker_color=COLORS["slate"],
            text=[f"{value:.1f} ms" for value in baseline_itl],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            name="decode-tail-1024",
            x=cohort_labels,
            y=candidate_itl,
            marker_color=COLORS["blue"],
            text=[f"{value:.1f} ms" for value in candidate_itl],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    for label, color in (
        ("ITL p99 improvement", COLORS["green"]),
        ("Long TTFT cost", COLORS["amber"]),
        ("Decode TPOT cost", COLORS["red"]),
        ("Goodput loss", COLORS["cyan"]),
    ):
        figure.add_trace(
            go.Bar(
                name=label,
                x=cohort_labels,
                y=metrics[label],
                marker_color=color,
                text=[f"{value:.3g}%" for value in metrics[label]],
                textposition="outside",
                cliponaxis=False,
                visible=True,
            ),
            row=1,
            col=2,
        )
    figure.update_yaxes(title_text="Milliseconds", range=[0, 185], row=1, col=1)
    figure.update_yaxes(title_text="Change vs production default (%)", range=[0, 48], row=1, col=2)
    _style(figure, "decode-tail-1024 — repeatable tail-latency gain within deployment guardrails")
    figure.add_annotation(
        text=(
            "12 runs / 6 profile pairs; 0 OOM / timeout / preemption; "
            "worst transient KV delta: 3 blocks (2.625 MiB)"
        ),
        x=0.5,
        y=-0.19,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": COLORS["slate"]},
    )
    return figure


def _write(figure: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        figure.write_image(path, width=1200, height=540, scale=2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate M6 PNG figures from sealed long-context v5 summaries"
    )
    parser.add_argument("--planner-artifact", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--apc-artifact", type=Path, default=DEFAULT_APC)
    parser.add_argument("--m5-artifact", type=Path, default=DEFAULT_M5)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load sealed summaries and render the three public M6 figures."""
    args = _parser().parse_args(argv)
    figures = {
        "longctx-v5-capacity-planner.png": _planner_figure(_load_summary(args.planner_artifact)),
        "longctx-v5-apc.png": _apc_figure(_load_summary(args.apc_artifact)),
        "longctx-v5-decode-tail.png": _m5_figure(_load_summary(args.m5_artifact)),
    }
    for name, figure in figures.items():
        path = args.output_dir / name
        _write(figure, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
