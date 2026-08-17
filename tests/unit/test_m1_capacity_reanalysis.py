"""End-to-end coverage for sealed, zero-GPU M1 capacity reanalysis."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_tuner.experiment.manifest import sha256_file
from vllm_tuner.longctx.m1_capacity_analysis import analyze_capacity_sweep
from vllm_tuner.longctx.m1_capacity_integrity import (
    M1_CAPACITY_INTEGRITY_FILE,
    seal_m1_capacity_artifacts,
    validate_m1_capacity_artifacts,
)
from vllm_tuner.longctx.m1_capacity_reanalysis import M1CapacityBoundaryRunner

from .test_m1_capacity_analysis import _policy, _valid_sweep


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _formal_source(root: Path, experiment_id: str) -> Path:
    source_root = root / experiment_id
    source_root.mkdir()
    trials = []
    for context_id, context_tokens, mode in (
        ("context-8k", 8_192, "joint"),
        ("context-16k", 16_384, "transitional"),
        ("context-32k", 32_768, "left-censored"),
    ):
        for trial in _valid_sweep():
            updates: dict[str, object] = {
                "trial_id": f"{context_id}-{trial.load_id}-r{trial.repeat_index}",
                "context_id": context_id,
                "context_tokens": context_tokens,
                "trace_id": f"{context_id}-{trial.load_id}-trace",
            }
            if mode == "transitional" and trial.load_id == "mid":
                updates.update(
                    goodput_requests_per_second=1.0,
                    slo_satisfied_fraction=0.50,
                )
            if mode == "left-censored" and trial.load_id in {"low", "mid"}:
                updates.update(
                    goodput_requests_per_second=0.20,
                    slo_satisfied_fraction=0.50,
                )
            trials.append(trial.model_copy(update=updates))
    analysis = analyze_capacity_sweep(trials, _policy())
    assert analysis.passed is False

    _write_json(
        source_root / "manifest.json",
        {
            "experiment_id": experiment_id,
            "source_commit": "source-commit",
            "source_tree_sha256": "source-tree",
        },
    )
    _write_json(
        source_root / "experiment.json",
        {
            "knee_policy": {
                "below_lowest_result": "left-censored-below-lowest-load",
                "no_overload_result": "right-censored-above-highest-load",
            }
        },
    )
    _write_json(
        source_root / "summary.json",
        {
            "schema_version": "longctx-m1-capacity.v1",
            "project_line": "longctx-v5",
            "milestone": "M1",
            "experiment_kind": "capacity-sweep",
            "evidence_role": "formal",
            "experiment_id": experiment_id,
            "execution": {
                "passed": True,
                "planned_jobs": 27,
                "completed_jobs": 27,
                "failed_jobs": 0,
                "unsafe_cleanup": False,
            },
            "acceptance": {
                "eligible": True,
                "passed": False,
                "checks": {
                    "all_capacity_jobs_complete": True,
                    "capacity_knees_accepted": False,
                    "planner_validation_passed": True,
                },
                "failure_reasons": ["capacity_knees_accepted"],
            },
            "analysis": analysis.model_dump(mode="json"),
        },
    )
    seal_m1_capacity_artifacts(
        source_root,
        experiment_id,
        {
            "experiment_id": experiment_id,
            "project_line": "longctx-v5",
            "milestone": "M1",
            "experiment_kind": "capacity-sweep",
            "evidence_role": "formal",
            "source_commit": "source-commit",
            "capacity_accepted": False,
        },
    )
    return source_root


def test_boundary_runner_preserves_source_and_seals_zero_gpu_result(tmp_path: Path) -> None:
    source_id = "formal-source"
    output_id = "formal-source-boundaries-v2"
    source_root = _formal_source(tmp_path, source_id)
    source_seal = source_root / M1_CAPACITY_INTEGRITY_FILE
    source_digest = sha256_file(source_seal)
    runner = M1CapacityBoundaryRunner(
        tmp_path,
        source_id,
        output_id,
        repository=tmp_path,
        analysis_source_identity=("analysis-commit", "analysis-tree"),
    )

    summary = runner.run()

    assert summary["acceptance"]["passed"] is True
    assert summary["gpu_runs_executed"] == 0
    assert summary["source_v1_acceptance"]["passed"] is False
    assert sha256_file(source_seal) == source_digest
    sealed = validate_m1_capacity_artifacts(tmp_path / output_id)
    assert sealed["attestation"]["capacity_accepted"] is True
    assert sealed["attestation"]["gpu_runs_executed"] == 0

    resumed = M1CapacityBoundaryRunner(
        tmp_path,
        source_id,
        output_id,
        repository=tmp_path,
        resume=True,
        analysis_source_identity=("analysis-commit", "analysis-tree"),
    ).run()
    assert resumed == summary
    assert sha256_file(source_seal) == source_digest
