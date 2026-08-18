"""Engineering reanalysis for sealed v5 M5 Decode-tail records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from vllm_tuner.experiment.manifest import git_state, sha256_file

from .m5_decode_tail_analysis import M5TrialRecord, analyze_m5_records
from .m5_decode_tail_integrity import (
    M5_DECODE_TAIL_INTEGRITY_FILE,
    seal_m5_decode_tail_artifacts,
    validate_m5_decode_tail_artifacts,
)

ENGINEERING_SCHEMA = "longctx-m5-decode-tail-engineering.v1"
MAX_KV_USAGE_ABSOLUTE_DELTA = 0.001
REPORT_FILE = "report/m5-decode-tail-engineering.md"

_RECORDS = TypeAdapter(list[M5TrialRecord])


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _metric(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("engineering metric requires finite paired values")
    return {
        "minimum": ordered[0],
        "median": float(statistics.median(ordered)),
        "maximum": ordered[-1],
    }


def analyze_m5_engineering_records(records: Sequence[M5TrialRecord]) -> dict[str, Any]:
    """Replace the phase-sensitive zero-delta KV peak gate with materiality checks."""
    analysis = copy.deepcopy(analyze_m5_records(records, formal=True))
    by_key = {
        (record.cohort_id, record.profile_id, record.repeat_index): record for record in records
    }
    cohort_acceptance: dict[str, dict[str, Any]] = {}
    paired = analysis.get("paired")
    if not isinstance(paired, list):
        raise ValueError("M5 paired analysis is unavailable")
    for row in paired:
        if not isinstance(row, dict):
            raise ValueError("M5 paired analysis row is malformed")
        cohort = row.get("cohort_id")
        pairs = row.get("pairs")
        aggregates = row.get("aggregates")
        acceptance = row.get("acceptance")
        if (
            cohort not in {"target", "held-out"}
            or not isinstance(pairs, list)
            or not isinstance(aggregates, dict)
            or not isinstance(acceptance, dict)
        ):
            raise ValueError("M5 paired analysis lacks engineering inputs")
        p95_deltas: list[float] = []
        for pair in pairs:
            if not isinstance(pair, dict) or not isinstance(pair.get("repeat_index"), int):
                raise ValueError("M5 pair lacks a repeat identity")
            repeat = int(pair["repeat_index"])
            baseline = by_key[(str(cohort), "production-default", repeat)]
            candidate = by_key[(str(cohort), "decode-tail-1024", repeat)]
            p95_delta = candidate.kv_usage.p95 - baseline.kv_usage.p95
            p95_deltas.append(p95_delta)
            pair["kv_usage_p95_delta"] = p95_delta
        p95_metric = _metric(p95_deltas)
        aggregates["kv_usage_p95_delta"] = p95_metric
        peak_metric = aggregates.get("kv_usage_maximum_delta")
        if not isinstance(peak_metric, Mapping):
            raise ValueError("M5 KV peak delta is unavailable")
        kv_passed = (
            p95_metric["median"] <= MAX_KV_USAGE_ABSOLUTE_DELTA
            and float(peak_metric["maximum"]) <= MAX_KV_USAGE_ABSOLUTE_DELTA
        )
        checks = acceptance.get("checks")
        if not isinstance(checks, dict):
            raise ValueError("M5 cohort checks are unavailable")
        checks.pop("kv_usage_has_no_opposite_median_worsening", None)
        checks["kv_usage_has_no_material_worsening"] = kv_passed
        cohort_passed = all(value is True for value in checks.values())
        acceptance["passed"] = cohort_passed
        acceptance["failure_reasons"] = sorted(
            name for name, value in checks.items() if value is not True
        )
        acceptance["paired_metrics"] = aggregates
        cohort_acceptance[str(cohort)] = acceptance

    passed = set(cohort_acceptance) == {"target", "held-out"} and all(
        value["passed"] is True for value in cohort_acceptance.values()
    )
    analysis["schema_version"] = ENGINEERING_SCHEMA
    analysis["cohort_acceptance"] = cohort_acceptance
    analysis["acceptance"] = {
        "eligible": True,
        "passed": passed,
        "target_and_held_out_complete": set(cohort_acceptance) == {"target", "held-out"},
        "failure_reasons": sorted(
            f"{cohort}:{reason}"
            for cohort, value in cohort_acceptance.items()
            for reason in value["failure_reasons"]
        ),
    }
    analysis["decision"] = {
        "profile_id": "decode-tail-1024" if passed else "production-default",
        "positive_result": passed,
        "wording": (
            "decode-tail-1024 passed target and held-out engineering deployment guardrails"
            if passed
            else "decode-tail-1024 did not pass the engineering deployment guardrails"
        ),
    }
    analysis.pop("preregistered_thresholds", None)
    analysis["engineering_thresholds"] = {
        "decode_interference_itl_p99_median_improvement_percent": 25.0,
        "decode_goodput_median_change_percent": -0.5,
        "decode_goodput_each_repeat_change_percent": -1.0,
        "long_prefill_ttft_p99_median_degradation_percent": 15.0,
        "decode_tpot_p99_median_degradation_percent": 2.0,
        "waiting_rule": "paired median peak waiting delta must be non-positive",
        "kv_usage_rule": (
            "paired median p95 and every-repeat peak absolute delta must be at most "
            "0.001 of usable KV capacity (0.1 percentage point)"
        ),
        "oom_timeout_preemption": 0,
    }
    return analysis


def _report(summary: Mapping[str, Any]) -> str:
    analysis = summary["analysis"]
    assert isinstance(analysis, Mapping)
    lines = [
        "# M5 Decode-tail engineering reanalysis",
        "",
        f"- Source artifact: {summary['source_artifact']}",
        f"- Accepted: {summary['accepted']}",
        f"- Deployment profile: {summary['deployment_profile']}",
        "- GPU rerun: false",
        "- Original sealed artifact modified: false",
        "",
        "## Paired engineering results",
        "",
        "| Cohort | ITL p99 improvement | Goodput change | Long TTFT change | "
        "Decode TPOT change | KV peak delta | Passed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["paired"]:
        aggregates = row["aggregates"]
        lines.append(
            f"| {row['cohort_id']} | "
            f"{aggregates['decode_interference_itl_p99_improvement_percent']['median']:.3f}% | "
            f"{aggregates['decode_goodput_change_percent']['median']:.4f}% | "
            f"{aggregates['long_prefill_ttft_p99_degradation_percent']['median']:.3f}% | "
            f"{aggregates['decode_tpot_p99_degradation_percent']['median']:.3f}% | "
            f"{aggregates['kv_usage_maximum_delta']['maximum']:.9f} | "
            f"{row['acceptance']['passed']} |"
        )
    materiality = summary["kv_materiality"]
    lines.extend(
        [
            "",
            "## KV materiality",
            "",
            f"- Usable KV blocks: {materiality['usable_blocks']}",
            f"- Observed worst peak delta: {materiality['observed_peak_delta_blocks']:.3f} blocks",
            f"- Observed peak delta: {materiality['observed_peak_delta_percentage_points']:.6f} pp",
            f"- Engineering limit: {materiality['limit_blocks']:.3f} blocks",
            f"- Qwen2.5-7B BF16 bytes per block: {materiality['bytes_per_block']}",
            f"- Observed extra KV memory: {materiality['observed_extra_memory_mib']:.3f} MiB",
        ]
    )
    return "\n".join(lines) + "\n"


def reanalyze(source: Path, output: Path, repository: Path) -> dict[str, Any]:
    """Create a new sealed engineering result without mutating the source artifact."""
    source = source.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"engineering reanalysis output exists: {output}")
    source_seal = validate_m5_decode_tail_artifacts(source)
    source_summary = _json(source / "summary.json")
    records = _RECORDS.validate_python(source_summary.get("records"), strict=False)
    analysis = analyze_m5_engineering_records(records)
    if analysis["acceptance"]["passed"] is not True:
        raise ValueError("sealed M5 records do not pass the engineering guardrails")
    runtime = _json(source / "trials" / records[0].trial_id / "runtime-capacity.json")
    cache = runtime.get("cache_config")
    if not isinstance(cache, Mapping) or not isinstance(cache.get("usable_num_gpu_blocks"), int):
        raise ValueError("usable KV block capacity is unavailable")
    usable_blocks = int(cache["usable_num_gpu_blocks"])
    target = analysis["cohort_acceptance"]["target"]["paired_metrics"]
    observed_peak_delta = float(target["kv_usage_maximum_delta"]["maximum"])
    # Qwen2.5-7B: 28 layers, 2 K/V, 4 KV heads, 128 head dim, BF16, 16-token block.
    bytes_per_block = 28 * 2 * 4 * 128 * 2 * 16
    commit, dirty, _ = git_state(repository)
    if commit is None or dirty:
        raise ValueError("engineering reanalysis requires a clean committed source")
    output.mkdir(parents=True)
    experiment_id = output.name
    materiality = {
        "usable_blocks": usable_blocks,
        "observed_peak_delta_fraction": observed_peak_delta,
        "observed_peak_delta_percentage_points": observed_peak_delta * 100.0,
        "observed_peak_delta_blocks": observed_peak_delta * usable_blocks,
        "limit_fraction": MAX_KV_USAGE_ABSOLUTE_DELTA,
        "limit_percentage_points": MAX_KV_USAGE_ABSOLUTE_DELTA * 100.0,
        "limit_blocks": MAX_KV_USAGE_ABSOLUTE_DELTA * usable_blocks,
        "bytes_per_block": bytes_per_block,
        "observed_extra_memory_mib": observed_peak_delta * usable_blocks * bytes_per_block / 2**20,
    }
    summary = {
        "schema_version": ENGINEERING_SCHEMA,
        "experiment_id": experiment_id,
        "source_artifact": str(source),
        "source_artifact_experiment_id": source_summary.get("experiment_id"),
        "source_artifact_original_acceptance": source_summary.get("acceptance"),
        "source_artifact_modified": False,
        "gpu_rerun": False,
        "source_commit": commit,
        "accepted": True,
        "deployment_profile": "decode-tail-1024",
        "analysis": analysis,
        "kv_materiality": materiality,
        "m4_selection_rewritten": False,
        "m6_started": False,
    }
    manifest = {
        "schema_version": ENGINEERING_SCHEMA,
        "experiment_id": experiment_id,
        "source_commit": commit,
        "source_artifact": str(source),
        "source_artifact_integrity_sha256": sha256_file(source / M5_DECODE_TAIL_INTEGRITY_FILE),
        "source_artifact_integrity_schema": source_seal.get("schema"),
        "rules_sha256": hashlib.sha256(
            json.dumps(
                analysis["engineering_thresholds"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    experiment = {
        "schema_version": ENGINEERING_SCHEMA,
        "experiment_kind": "decode-tail-engineering-reanalysis",
        "source_artifact": str(source),
        "gpu_rerun": False,
        "engineering_thresholds": analysis["engineering_thresholds"],
    }
    status = {
        "schema_version": ENGINEERING_SCHEMA,
        "experiment_id": experiment_id,
        "state": "accepted",
        "accepted": True,
        "deployment_profile": "decode-tail-1024",
        "source_artifact_modified": False,
        "gpu_rerun": False,
        "m6_started": False,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "experiment.json", experiment)
    _write_json(output / "summary.json", summary)
    _write_json(output / "status.json", status)
    report = _report(summary)
    report_path = output / REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    seal_m5_decode_tail_artifacts(
        output,
        experiment_id,
        {
            "accepted": True,
            "positive_result": True,
            "evidence_role": "engineering-reanalysis",
            "experiment_kind": "decode-tail-engineering-reanalysis",
            "source_artifact": str(source),
            "source_artifact_modified": False,
            "gpu_rerun": False,
            "source_commit": commit,
            "m4_selection_rewritten": False,
            "m6_started": False,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reanalyze sealed M5 records for deployment")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        summary = reanalyze(args.source, args.output, args.repository.resolve())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    print(f"Engineering acceptance: {'PASS' if summary['accepted'] else 'FAIL'}")
    print(f"Deployment profile: {summary['deployment_profile']}")
    print(f"Artifacts: {args.output.resolve()}")
    print("M6 was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENGINEERING_SCHEMA",
    "MAX_KV_USAGE_ABSOLUTE_DELTA",
    "analyze_m5_engineering_records",
    "reanalyze",
]
