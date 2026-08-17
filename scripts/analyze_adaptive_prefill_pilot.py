#!/usr/bin/env python3
"""Build phase-level fixed-policy tradeoff and offline Oracle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_tuner.analysis.nonstationary import (
    aggregate_policy_trials,
    select_phase_oracle,
    summarize_labeled_requests,
)
from vllm_tuner.workloads.trace import WorkloadTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze phase labels offline; the runtime controller never sees them."
    )
    parser.add_argument("--trace", required=True, help="Frozen phase-labeled JSONL trace")
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="NAME=EXPERIMENT_DIR",
        help="Policy name and completed experiment directory; repeat for each policy",
    )
    parser.add_argument(
        "--fixed-policy",
        action="append",
        default=[],
        help="Policy eligible for the non-deployable per-phase Oracle",
    )
    parser.add_argument("--slo-ttft-ms", type=float, required=True)
    parser.add_argument("--slo-tpot-ms", type=float, required=True)
    parser.add_argument("--slo-e2e-ms", type=float, required=True)
    parser.add_argument("--output", help="Optional JSON output; stdout is always emitted")
    return parser.parse_args()


def _parse_policies(values: list[str]) -> dict[str, Path]:
    policies: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"policy must use NAME=EXPERIMENT_DIR: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in policies:
            raise ValueError(f"policy name must be non-empty and unique: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        policies[name] = path
    return policies


def _load_request_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _complete_measurement_trials(experiment: Path) -> list[Path]:
    trials: list[Path] = []
    for trial in sorted((experiment / "trials").iterdir()):
        if trial.name.startswith(("capacity-", "holdout-")):
            continue
        summary_path = trial / "summary.json"
        requests_path = trial / "request-results.jsonl"
        if not summary_path.is_file() or not requests_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            continue
        if summary.get("constraints", {}).get("feasible") is not True:
            continue
        trials.append(trial)
    if not trials:
        raise ValueError(f"no complete feasible measurement trials in {experiment}")
    return trials


def main() -> int:
    args = parse_args()
    policies = _parse_policies(args.policy)
    trace_path = Path(args.trace).expanduser().resolve()
    trace = WorkloadTrace.read(
        trace_path,
        seed=0,
        profile="nonstationary",
        request_rate=None,
        burstiness=1.0,
    )
    slo = {
        "ttft_ms": args.slo_ttft_ms,
        "tpot_ms": args.slo_tpot_ms,
        "e2e_ms": args.slo_e2e_ms,
    }
    aggregates: dict[str, dict[str, Any]] = {}
    trial_ids: dict[str, list[str]] = {}
    for name, experiment in policies.items():
        trials = _complete_measurement_trials(experiment)
        trial_ids[name] = [trial.name for trial in trials]
        summaries = [
            summarize_labeled_requests(
                trace,
                _load_request_rows(trial / "request-results.jsonl"),
                slo=slo,
            )
            for trial in trials
        ]
        aggregates[name] = aggregate_policy_trials(summaries)

    oracle = (
        select_phase_oracle(aggregates, eligible_policies=args.fixed_policy)
        if args.fixed_policy
        else None
    )
    payload = {
        "schema_version": 1,
        "trace": str(trace_path),
        "trace_sha256": trace.checksum(),
        "slo_ms": slo,
        "phase_labels_visible_to_runtime": False,
        "policy_trial_ids": trial_ids,
        "policies": aggregates,
        "offline_per_phase_oracle": oracle,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
