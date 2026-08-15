#!/usr/bin/env python3
"""Run deterministic fixed-token-budget versus adaptive scheduler ablations."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

from vllm_tuner.experiment.manifest import (
    collect_environment_fingerprint,
    git_state,
    sha256_json,
    source_tree_sha256,
)
from vllm_tuner.scheduling import (
    DEFAULT_FIXED_BUDGETS,
    AdaptiveBudgetConfig,
    AdmissionConfig,
    BudgetAblationReport,
    SimulationConfig,
    SimulationRequest,
    TraceComparison,
    run_budget_ablation,
)

JSON_ARTIFACT_NAME = "scheduler_ablation.json"
MARKDOWN_ARTIFACT_NAME = "scheduler_ablation.md"
_REQUEST_FIELDS = {
    "request_id",
    "arrival_time",
    "prompt_tokens",
    "output_tokens",
    "priority",
    "ttft_slo",
    "tpot_slo",
    "e2e_slo",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic SLOTune scheduler ablations with fixed token budgets "
            "and the adaptive policy. An explicit output directory is required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
JSONL request schema (one object per line):
  {"request_id":"r0","arrival_time":0.0,"prompt_tokens":256,"output_tokens":64}

A combined --trace-jsonl also requires "split":"calibration" or "split":"held_out".
Separate trace files use the same schema without the split field.

Examples:
  python scripts/run_scheduler_ablation.py \\
    --output-dir /root/autodl-tmp/scheduler-ablation/builtin

  python scripts/run_scheduler_ablation.py \\
    --trace-jsonl traces/mixed_scheduler.jsonl \\
    --output-dir /root/autodl-tmp/scheduler-ablation/mixed \\
    --fixed-budgets 512 1024 2048 4096 8192

  python scripts/run_scheduler_ablation.py \\
    --calibration-trace-jsonl traces/calibration.jsonl \\
    --held-out-trace-jsonl traces/held_out.jsonl \\
    --output-dir /root/autodl-tmp/scheduler-ablation/pair \\
    --overwrite
""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Explicit artifact directory; writes scheduler_ablation.json and "
            "scheduler_ablation.md"
        ),
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        help="Combined fixed trace with a calibration/held_out split on every line",
    )
    parser.add_argument(
        "--calibration-trace-jsonl",
        type=Path,
        help="Calibration JSONL trace (must be paired with --held-out-trace-jsonl)",
    )
    parser.add_argument(
        "--held-out-trace-jsonl",
        type=Path,
        help="Held-out JSONL trace (must be paired with --calibration-trace-jsonl)",
    )
    parser.add_argument(
        "--fixed-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_FIXED_BUDGETS),
        metavar="TOKENS",
        help="Two or more fixed baselines (default: 512 1024 2048 4096 8192)",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Recorded deterministic seed")
    parser.add_argument(
        "--step-duration",
        type=float,
        default=0.01,
        help="Seconds represented by one simulator step (default: 0.01)",
    )
    parser.add_argument(
        "--prefill-quantum",
        type=int,
        default=128,
        help="Maximum prefill chunk per request and round (default: 128)",
    )
    parser.add_argument(
        "--kv-capacity-tokens",
        type=int,
        default=65536,
        help="Synthetic KV capacity used to calculate pressure (default: 65536)",
    )
    parser.add_argument(
        "--available-token-budget",
        type=int,
        help="Optional temporary cap applied to both fixed and adaptive decisions",
    )
    parser.add_argument("--ttft-slo", type=float, default=1.0, help="TTFT SLO in seconds")
    parser.add_argument("--tpot-slo", type=float, default=0.05, help="TPOT SLO in seconds")
    parser.add_argument("--e2e-slo", type=float, default=10.0, help="E2E SLO in seconds")
    parser.add_argument(
        "--starvation-threshold",
        type=float,
        default=2.0,
        help="Maximum service gap before a request is counted as starved",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=1.0,
        help="Aging/max-wait threshold in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--minimum-prefill-progress",
        type=int,
        default=32,
        help="Reserved prefill tokens when prefill is waiting (default: 32)",
    )
    parser.add_argument(
        "--max-admitted-sequences",
        type=int,
        default=64,
        help="Maximum admitted sequences for fixed/adaptive comparisons",
    )
    parser.add_argument("--adaptive-min-budget", type=int, default=512, help="Adaptive lower bound")
    parser.add_argument(
        "--adaptive-max-budget", type=int, default=8192, help="Adaptive upper bound"
    )
    parser.add_argument(
        "--adaptive-initial-budget",
        type=int,
        default=2048,
        help="Adaptive initial budget",
    )
    parser.add_argument(
        "--adaptive-budget-step", type=int, default=512, help="Adaptive adjustment size"
    )
    parser.add_argument(
        "--adaptive-hysteresis-steps",
        type=int,
        default=2,
        help="Consecutive direction samples required before a budget change",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing scheduler_ablation.json/.md in the explicit output directory",
    )
    return parser


def builtin_traces() -> tuple[tuple[SimulationRequest, ...], tuple[SimulationRequest, ...]]:
    """Return deterministic mixed calibration and structurally different holdout traces."""

    calibration: list[SimulationRequest] = []
    for index in range(8):
        calibration.append(
            SimulationRequest(
                request_id=f"cal-chat-{index}",
                arrival_time=index * 0.015,
                prompt_tokens=192 + (index % 3) * 64,
                output_tokens=24 + (index % 2) * 8,
            )
        )
    calibration.extend(
        [
            SimulationRequest("cal-rag-0", 0.020, 2048, 16),
            SimulationRequest("cal-rag-1", 0.065, 4096, 16),
            SimulationRequest("cal-rag-2", 0.110, 3072, 24),
        ]
    )

    held_out: list[SimulationRequest] = []
    for index in range(6):
        held_out.append(
            SimulationRequest(
                request_id=f"hold-chat-{index}",
                arrival_time=(index // 3) * 0.08 + (index % 3) * 0.002,
                prompt_tokens=96 + index * 48,
                output_tokens=32,
            )
        )
    held_out.extend(
        [
            SimulationRequest("hold-rag-0", 0.001, 3584, 12),
            SimulationRequest("hold-rag-1", 0.081, 5120, 20),
        ]
    )
    return tuple(calibration), tuple(held_out)


def _request_from_record(record: dict[str, Any], path: Path, line_number: int) -> SimulationRequest:
    unknown = sorted(set(record) - _REQUEST_FIELDS - {"split"})
    if unknown:
        raise ValueError(f"{path}:{line_number}: unknown request fields: {unknown}")
    missing = sorted({"request_id", "arrival_time", "prompt_tokens", "output_tokens"} - set(record))
    if missing:
        raise ValueError(f"{path}:{line_number}: missing request fields: {missing}")

    request_id = record["request_id"]
    arrival_time = record["arrival_time"]
    prompt_tokens = record["prompt_tokens"]
    output_tokens = record["output_tokens"]
    priority = record.get("priority", 0)
    if not isinstance(request_id, str):
        raise ValueError(f"{path}:{line_number}: request_id must be a string")
    if isinstance(arrival_time, bool) or not isinstance(arrival_time, (int, float)):
        raise ValueError(f"{path}:{line_number}: arrival_time must be numeric")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise ValueError(f"{path}:{line_number}: prompt_tokens must be an integer")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
        raise ValueError(f"{path}:{line_number}: output_tokens must be an integer")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError(f"{path}:{line_number}: priority must be an integer")

    optional_slos: dict[str, Optional[float]] = {}
    for key in ("ttft_slo", "tpot_slo", "e2e_slo"):
        value = record.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{path}:{line_number}: {key} must be numeric or null")
        optional_slos[key] = None if value is None else float(value)
    try:
        return SimulationRequest(
            request_id=request_id,
            arrival_time=float(arrival_time),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            priority=priority,
            ttft_slo=optional_slos["ttft_slo"],
            tpot_slo=optional_slos["tpot_slo"],
            e2e_slo=optional_slos["e2e_slo"],
        )
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error


def _read_jsonl_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read trace {path}: {error}") from error
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
        records.append((line_number, value))
    if not records:
        raise ValueError(f"trace {path} contains no requests")
    return records


def load_trace_jsonl(path: Path) -> tuple[SimulationRequest, ...]:
    """Load one fixed trace; an optional split field is ignored."""

    return tuple(
        _request_from_record(record, path, line_number)
        for line_number, record in _read_jsonl_records(path)
    )


def load_split_trace_jsonl(
    path: Path,
) -> tuple[tuple[SimulationRequest, ...], tuple[SimulationRequest, ...]]:
    """Load a combined JSONL trace partitioned by its explicit split field."""

    calibration: list[SimulationRequest] = []
    held_out: list[SimulationRequest] = []
    for line_number, record in _read_jsonl_records(path):
        split = record.get("split")
        if split == "holdout":
            split = "held_out"
        if split not in {"calibration", "held_out"}:
            raise ValueError(f"{path}:{line_number}: split must be 'calibration' or 'held_out'")
        request = _request_from_record(record, path, line_number)
        if split == "calibration":
            calibration.append(request)
        else:
            held_out.append(request)
    if not calibration or not held_out:
        raise ValueError(f"combined trace {path} must contain both calibration and held_out")
    return tuple(calibration), tuple(held_out)


def _select_traces(
    args: argparse.Namespace,
) -> tuple[tuple[SimulationRequest, ...], tuple[SimulationRequest, ...], str]:
    paired_any = args.calibration_trace_jsonl or args.held_out_trace_jsonl
    if args.trace_jsonl and paired_any:
        raise ValueError(
            "--trace-jsonl cannot be combined with separate calibration/held-out traces"
        )
    if bool(args.calibration_trace_jsonl) != bool(args.held_out_trace_jsonl):
        raise ValueError(
            "--calibration-trace-jsonl and --held-out-trace-jsonl must be provided together"
        )
    if args.trace_jsonl:
        calibration, held_out = load_split_trace_jsonl(args.trace_jsonl)
        return calibration, held_out, f"combined_jsonl:{args.trace_jsonl}"
    if paired_any:
        calibration = load_trace_jsonl(args.calibration_trace_jsonl)
        held_out = load_trace_jsonl(args.held_out_trace_jsonl)
        source = f"paired_jsonl:{args.calibration_trace_jsonl},{args.held_out_trace_jsonl}"
        return calibration, held_out, source
    calibration, held_out = builtin_traces()
    return calibration, held_out, "builtin_deterministic_v1"


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _comparison_markdown(comparison: TraceComparison) -> list[str]:
    lines = [
        f"## {comparison.trace_name}",
        "",
        "| Policy | Goodput req/s | p50 queue | p99 queue | p50 TTFT | p99 TTFT | "
        "p50 TPOT | p99 TPOT | Fairness | Starvation | Preemptions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [(f"fixed-{budget}", result) for budget, result in comparison.fixed_baselines.items()]
    rows.append(("adaptive", comparison.adaptive))
    for policy_name, result in rows:
        metrics = result.metrics
        lines.append(
            f"| {policy_name} | {_format_float(metrics.goodput)} | "
            f"{_format_float(metrics.p50_queue_time)} | "
            f"{_format_float(metrics.p99_queue_time)} | "
            f"{_format_float(metrics.p50_ttft)} | "
            f"{_format_float(metrics.p99_ttft)} | "
            f"{_format_float(metrics.p50_tpot)} | "
            f"{_format_float(metrics.p99_tpot)} | "
            f"{_format_float(metrics.fairness_index)} | "
            f"{metrics.starvation_count} | {metrics.preemption_count} |"
        )
    lines.extend(
        [
            "",
            f"Best fixed budget by goodput: **{comparison.best_fixed_budget}**",
            "",
            "Adaptive goodput gain versus that baseline: "
            f"**{comparison.goodput_gain_vs_best * 100.0:.2f}%**",
            "",
        ]
    )
    return lines


def render_markdown(
    report: BudgetAblationReport,
    trace_source: str,
    fixed_budgets: Sequence[int],
    provenance: dict[str, Any],
    trace_sha256: dict[str, str],
) -> str:
    """Render a compact human-readable ablation report."""

    lines = [
        "# SLOTune deterministic scheduler ablation",
        "",
        f"Trace source: `{trace_source}`",
        "",
        f"Source commit: `{provenance['source_commit']}`",
        "",
        f"Source tree SHA-256: `{provenance['source_tree_sha256']}`",
        "",
        f"Dirty worktree: `{provenance['dirty_worktree']}`",
        "",
        f"Calibration trace SHA-256: `{trace_sha256['calibration']}`",
        "",
        f"Held-out trace SHA-256: `{trace_sha256['held_out']}`",
        "",
        "Fixed budgets: " + ", ".join(str(budget) for budget in fixed_budgets),
        "",
    ]
    lines.extend(_comparison_markdown(report.calibration))
    lines.extend(_comparison_markdown(report.held_out))
    lines.extend(
        [
            "## Negative or no-benefit conditions",
            "",
        ]
    )
    if report.negative_gain_conditions:
        lines.extend(
            [
                "| Trace | Metric | Adaptive | Fixed | Budget | Relative gain | Explanation |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for condition in report.negative_gain_conditions:
            lines.append(
                f"| {condition.trace_name} | {condition.metric} | "
                f"{_format_float(condition.adaptive_value)} | "
                f"{_format_float(condition.fixed_value)} | {condition.fixed_budget} | "
                f"{condition.relative_gain * 100.0:.2f}% | {condition.explanation} |"
            )
    else:
        lines.append("No negative/no-benefit condition was detected for these two traces.")
    lines.append("")
    return "\n".join(lines)


def _write_artifacts(
    output_dir: Path,
    payload: dict[str, Any],
    markdown: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    json_path = output_dir / JSON_ARTIFACT_NAME
    markdown_path = output_dir / MARKDOWN_ARTIFACT_NAME
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise ValueError(f"refusing to overwrite existing artifacts: {joined}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, run both traces, and write deterministic artifacts."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        calibration, held_out, trace_source = _select_traces(args)
        if len(args.fixed_budgets) < 2:
            raise ValueError("--fixed-budgets requires at least two values")
        if len(args.fixed_budgets) != len(set(args.fixed_budgets)):
            raise ValueError("--fixed-budgets values must be unique")

        simulation_config = SimulationConfig(
            step_duration=args.step_duration,
            prefill_quantum=args.prefill_quantum,
            kv_capacity_tokens=args.kv_capacity_tokens,
            available_token_budget=args.available_token_budget,
            seed=args.seed,
            ttft_slo=args.ttft_slo,
            tpot_slo=args.tpot_slo,
            e2e_slo=args.e2e_slo,
            starvation_threshold=args.starvation_threshold,
        )
        adaptive_config = AdaptiveBudgetConfig(
            min_budget=args.adaptive_min_budget,
            max_budget=args.adaptive_max_budget,
            initial_budget=args.adaptive_initial_budget,
            budget_step=args.adaptive_budget_step,
            hysteresis_steps=args.adaptive_hysteresis_steps,
            max_wait=args.max_wait,
            minimum_prefill_progress=args.minimum_prefill_progress,
            max_admitted_sequences=args.max_admitted_sequences,
        )
        admission_config = AdmissionConfig(
            max_wait=args.max_wait,
            minimum_prefill_progress=args.minimum_prefill_progress,
        )
        report = run_budget_ablation(
            calibration_trace=calibration,
            held_out_trace=held_out,
            fixed_budgets=args.fixed_budgets,
            adaptive_config=adaptive_config,
            simulation_config=simulation_config,
            admission_config=admission_config,
        )
        repository = Path(__file__).resolve().parents[1]
        source_commit, dirty_worktree, _ = git_state(repository)
        provenance = {
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256(repository),
            "dirty_worktree": dirty_worktree,
            "environment": collect_environment_fingerprint().model_dump(mode="json"),
        }
        trace_sha256 = {
            "calibration": sha256_json([asdict(request) for request in calibration]),
            "held_out": sha256_json([asdict(request) for request in held_out]),
        }
        payload = {
            "schema_version": 2,
            "provenance": provenance,
            "trace_source": trace_source,
            "trace_sha256": trace_sha256,
            "trace_sizes": {
                "calibration": len(calibration),
                "held_out": len(held_out),
            },
            "fixed_budgets": list(args.fixed_budgets),
            "simulation_config": asdict(simulation_config),
            "adaptive_config": asdict(adaptive_config),
            "admission_config": asdict(admission_config),
            "negative_gain_conditions": [
                condition.to_dict() for condition in report.negative_gain_conditions
            ],
            "report": report.to_dict(),
        }
        markdown = render_markdown(
            report,
            trace_source,
            args.fixed_budgets,
            provenance,
            trace_sha256,
        )
        json_path, markdown_path = _write_artifacts(
            args.output_dir, payload, markdown, args.overwrite
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"JSON artifact: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
