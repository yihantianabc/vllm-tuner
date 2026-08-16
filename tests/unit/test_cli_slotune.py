"""CLI coverage for immutable SLOTune experiment artifacts."""

import json
from pathlib import Path

import pandas as pd
import yaml
from click.utils import strip_ansi
from typer.main import get_command
from typer.testing import CliRunner

from vllm_tuner.cli.main import app
from vllm_tuner.experiment.artifacts import SUMMARY_COMPACT_FILE, ArtifactStore
from vllm_tuner.experiment.manifest import sha256_file
from vllm_tuner.experiment.models import TrialResult, TrialStatus
from vllm_tuner.reporting.export import export_best_config

runner = CliRunner()


def _write_experiment(results_root: Path, experiment_id: str = "exp") -> Path:
    experiment = results_root / experiment_id
    aggregate = experiment / "aggregate"
    report = experiment / "report"
    aggregate.mkdir(parents=True)
    report.mkdir()
    parameters = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    best = {
        "method": "default",
        "status": "COMPLETE",
        "feasible": True,
        "validated": True,
        "candidate": "default-0",
        "metric_provenance": "median_of_complete_feasible_repeats",
        "search_observation": {
            "trial_id": "default-0000",
            "trial_number": 0,
            "method": "default",
            "parameters": parameters,
            "goodput_requests_per_sec": 9.9,
        },
        "repeat_required": 3,
        "repeat_complete_feasible": 3,
        "holdout_required": True,
        "holdout_complete_feasible": 3,
        "repeat_metrics": {
            "goodput_requests_per_sec": {
                "count": 3,
                "median": 3.5,
                "min": 3.4,
                "max": 3.6,
            },
            "achieved_requests_per_sec": {
                "count": 3,
                "median": 3.8,
                "min": 3.7,
                "max": 3.9,
            },
            "p99_ttft_ms": {
                "count": 3,
                "median": 24.0,
                "min": 23.0,
                "max": 25.0,
            },
        },
        "holdout_metrics": {
            "goodput_requests_per_sec": {
                "count": 3,
                "median": 3.3,
                "min": 3.2,
                "max": 3.4,
            }
        },
        "parameters": parameters,
        "goodput_requests_per_sec": 3.5,
        "offered_requests_per_sec": 4.0,
        "achieved_requests_per_sec": 3.8,
        "p99_ttft_ms": 24.0,
        "p99_tpot_ms": 3.0,
        "p99_e2e_ms": 80.0,
    }
    repetitions = [
        {
            "trial_id": f"repeat-default-0-{repeat}",
            "trial_number": repeat,
            "method": "default",
            "parameters": parameters,
            "repeat_of": 0,
            "holdout": False,
            "status": "COMPLETE",
            "feasible": True,
            "goodput_requests_per_sec": goodput,
        }
        for repeat, goodput in enumerate((3.4, 3.5, 3.6))
    ]
    holdouts = [
        {
            "trial_id": f"holdout-default-0-{repeat}",
            "trial_number": repeat,
            "method": "default",
            "parameters": parameters,
            "repeat_of": 0,
            "holdout": True,
            "status": "COMPLETE",
            "feasible": True,
            "goodput_requests_per_sec": goodput,
        }
        for repeat, goodput in enumerate((3.2, 3.3, 3.4))
    ]
    manifest = {
        "experiment_id": experiment_id,
        "created_at": "2026-08-15T00:00:00+00:00",
        "model": "fake/model",
        "model_revision": "revision-1",
        "tokenizer": "fake/tokenizer",
        "vllm_args": {"max-model-len": 4096},
        "source_commit": "abc123",
        "trace_sha256": "search-trace",
        "holdout_trace_sha256": "holdout-trace",
        "artifact_warnings": [],
    }
    summary = {
        "experiment_id": experiment_id,
        "manifest": manifest,
        "search": {"default": [best]},
        "best": best,
        "candidate_validation": [
            {
                "candidate": "default-0",
                "method": "default",
                "parameters": parameters,
                "validated": True,
            }
        ],
        "repetitions": repetitions,
        "holdout": holdouts,
        "capacity_sweep": {"points": [], "by_rate": []},
        "report": {
            "html": str(report / "report.html"),
            "markdown": str(report / "report.md"),
        },
    }
    (experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (report / "report.html").write_text("<html>SLOTune static report</html>", encoding="utf-8")
    (report / "report.md").write_text("# Existing detailed report\n", encoding="utf-8")
    pd.DataFrame([best]).to_parquet(aggregate / "trials.parquet", index=False)
    pd.DataFrame(holdouts).to_parquet(aggregate / "holdout-results.parquet", index=False)
    return experiment


def _write_attestable_experiment(results_root: Path) -> tuple[Path, ArtifactStore]:
    experiment = _write_experiment(results_root)
    store = ArtifactStore(results_root, "exp")
    store.write_yaml("experiment.yaml", {"model": "fake/model"})
    trace_path = store.write_text("trace.jsonl", '{"request_id":"search"}\n')
    holdout_path = store.write_text("holdout-trace.jsonl", '{"request_id":"holdout"}\n')
    trace_sha256 = sha256_file(trace_path)
    holdout_sha256 = sha256_file(holdout_path)
    store.write_text("trace.sha256", f"{trace_sha256}  trace.jsonl\n")
    store.write_text("holdout-trace.sha256", f"{holdout_sha256}  holdout-trace.jsonl\n")
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = trace_sha256
    manifest["holdout_trace_sha256"] = holdout_sha256
    store.write_json("manifest.json", manifest)
    metrics = {
        "goodput": 1.0,
        "p99_ttft": 0.1,
        "p99_tpot": 0.01,
        "preemption_count": 0,
    }
    simulation = {
        "policy_name": "adaptive",
        "seed": 1,
        "metrics": metrics,
        "requests": [{"request_id": "scheduler"}],
        "steps": [{"step": 1}],
        "decisions": [{"budget": 512}],
    }
    section = {
        "trace_name": "calibration",
        "adaptive": simulation,
        "fixed_baselines": {},
        "best_fixed_budget": None,
        "goodput_gain_vs_best": 0.0,
        "negative_gain_conditions": [],
    }
    scheduler = {
        "calibration": section,
        "held_out": {**section, "trace_name": "held_out"},
        "has_negative_result": False,
        "negative_gain_conditions": [],
    }
    store.write_json("aggregate/scheduler-ablation.json", scheduler)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest"] = manifest
    summary["scheduler_ablation"] = scheduler
    store.write_json("summary.json", summary)
    store.write_json("report/plot-manifest.json", {"schema_version": 1, "plots": {}})
    failed = TrialResult(
        trial_id="failed-0",
        method="default",
        status=TrialStatus.FAILED,
        params={"max_num_seqs": 8},
        constraints={"feasible": False, "violations": ["test_failure"]},
        failure_reason={"type": "TEST_FAILURE", "message": "expected"},
    )
    store.ensure_trial_artifacts(failed)
    return experiment, store


def test_report_reuses_or_copies_existing_static_html(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    reused = runner.invoke(
        app,
        ["report", "-n", "exp", "--results-root", str(tmp_path), "-f", "html"],
    )
    assert reused.exit_code == 0, reused.output
    assert f"HTML report reused: {experiment / 'report/report.html'}" in reused.output

    copied_path = tmp_path / "copied.html"
    copied = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "html",
            "-o",
            str(copied_path),
        ],
    )
    assert copied.exit_code == 0, copied.output
    assert copied_path.read_text() == "<html>SLOTune static report</html>"


def test_attest_is_idempotent_requires_valid_existing_seal_and_explicit_reseal(
    tmp_path: Path,
) -> None:
    experiment, store = _write_attestable_experiment(tmp_path)
    raw = experiment / "aggregate/scheduler-ablation.json"
    anchor = experiment / "trials/failed-0/artifact-integrity.json"
    raw_sha256 = sha256_file(raw)
    anchor_sha256 = sha256_file(anchor)
    summary = experiment / "summary.json"
    summary_sha256 = sha256_file(summary)
    summary_size = summary.stat().st_size

    first = runner.invoke(
        app,
        ["attest", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert first.exit_code == 0, first.output
    assert "1/1 trials validated" in first.output
    seal = experiment / "experiment-integrity.json"
    first_seal = sha256_file(seal)
    assert sha256_file(raw) == raw_sha256
    assert sha256_file(anchor) == anchor_sha256
    assert sha256_file(summary) == summary_sha256
    assert summary.stat().st_size == summary_size
    compact = json.loads((experiment / SUMMARY_COMPACT_FILE).read_text())
    assert compact["experiment_attestation"]["original_summary"] == {
        "path": "summary.json",
        "size_bytes": summary_size,
        "sha256": summary_sha256,
    }
    integrity = json.loads(seal.read_text())
    record = integrity["attestations"][-1]
    assert record["attestation_kind"] == "cli-post-run-attestation"
    assert record["attested_at_utc"]
    assert "attestation_source_commit" in record
    assert record["attestation_source_tree_sha256"]
    store.validate_experiment_integrity()

    repeated = runner.invoke(
        app,
        ["attest", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert repeated.exit_code == 0, repeated.output
    assert "valid and unchanged" in repeated.output
    assert sha256_file(seal) == first_seal

    resealed = runner.invoke(
        app,
        ["attest", "-n", "exp", "--results-root", str(tmp_path), "--reseal"],
    )
    assert resealed.exit_code == 0, resealed.output
    assert sha256_file(seal) != first_seal
    assert sha256_file(raw) == raw_sha256
    assert sha256_file(anchor) == anchor_sha256
    assert sha256_file(summary) == summary_sha256
    assert summary.stat().st_size == summary_size
    store.validate_experiment_integrity()

    store.write_text("report/report.md", "corrupted after seal\n")
    corrupt = runner.invoke(
        app,
        ["attest", "-n", "exp", "--results-root", str(tmp_path), "--reseal"],
    )
    assert corrupt.exit_code == 1
    assert "artifact checksum mismatch: report/report.md" in corrupt.output


def test_cli_exports_cannot_overwrite_raw_or_identity_artifacts(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    raw_path = experiment / "aggregate/trials.parquet"
    raw_sha256 = sha256_file(raw_path)

    report_result = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "json",
            "-o",
            str(raw_path),
        ],
    )
    assert report_result.exit_code == 1
    assert "cannot be overwritten" in report_result.output
    assert sha256_file(raw_path) == raw_sha256

    export_result = runner.invoke(
        app,
        [
            "export",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-o",
            str(experiment / "summary.json"),
        ],
    )
    assert export_result.exit_code == 1
    assert "must be exactly best.yaml or best.json" in export_result.output


def test_report_exports_current_summary_and_aggregate_parquet(tmp_path: Path) -> None:
    _write_experiment(tmp_path)
    json_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "json",
            "-o",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "slotune-experiment-report-v1"
    assert payload["summary"]["best"]["parameters"]["tensor_parallel_size"] == 1
    assert payload["aggregates"]["trials"]["available"] is True
    assert payload["aggregates"]["trials"]["row_count"] == 1
    assert payload["aggregates"]["capacity_sweep"]["available"] is False
    assert payload["aggregates"]["capacity_sweep"]["records"] == []


def test_report_markdown_uses_slotune_metrics_and_validation(tmp_path: Path) -> None:
    _write_experiment(tmp_path)
    markdown_path = tmp_path / "summary.md"
    result = runner.invoke(
        app,
        [
            "report",
            "-n",
            "exp",
            "--results-root",
            str(tmp_path),
            "-f",
            "markdown",
            "-o",
            str(markdown_path),
        ],
    )
    assert result.exit_code == 0, result.output
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# SLOTune experiment: exp" in markdown
    assert "SLO goodput, repeat median (range): `3.500 (3.400–3.600; n=3)`" in markdown
    assert "Search SLO goodput (selection observation only): `9.900`" in markdown
    assert "Holdout validation: `passed`" in markdown
    assert "tensor_parallel_size" in markdown
    assert "Average Latency" not in markdown


def test_export_reads_best_parameters_after_exact_holdout_validation(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    result = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    exported = yaml.safe_load((experiment / "best.yaml").read_text())
    assert exported["source_experiment"] == "exp"
    assert exported["model"] == "fake/model"
    assert exported["model_revision"] == "revision-1"
    assert exported["tokenizer"] == "fake/tokenizer"
    assert exported["vllm_params"] == {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    assert exported["base_vllm_args"] == {"max-model-len": 4096}
    assert exported["vllm_args"] == {
        "max-model-len": 4096,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    assert exported["performance_metrics"]["goodput_requests_per_sec"] == 3.5
    assert exported["validation"]["validated"] is True
    assert exported["validation"]["matching_repeat_trial_ids"] == [
        "repeat-default-0-0",
        "repeat-default-0-1",
        "repeat-default-0-2",
    ]
    assert exported["validation"]["matching_holdout_trial_ids"] == [
        "holdout-default-0-0",
        "holdout-default-0-1",
        "holdout-default-0-2",
    ]


def test_export_rejects_unvalidated_or_empty_best(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["best"]["validated"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    unvalidated = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert unvalidated.exit_code == 1
    assert "summary.best.validated is not true" in unvalidated.output

    summary["best"] = None
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    empty = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )
    assert empty.exit_code == 1
    assert "summary.best is empty" in empty.output


def test_export_rejects_same_parameters_from_a_different_method(tmp_path: Path) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    for row in summary["repetitions"]:
        row["method"] = "random"
    for row in summary["holdout"]:
        row["method"] = "random"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "every exact repeat and holdout row" in result.output
    assert not (experiment / "best.yaml").exists()


def test_export_rejects_conflicting_manifest_and_validated_parameters(
    tmp_path: Path,
) -> None:
    experiment = _write_experiment(tmp_path)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest"]["vllm_args"]["tensor-parallel-size"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = runner.invoke(
        app,
        ["export", "-n", "exp", "--results-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "conflict with manifest.vllm_args" in result.output
    assert not (experiment / "best.yaml").exists()


def test_export_helper_retains_explicit_legacy_compatibility(tmp_path: Path) -> None:
    output = tmp_path / "legacy.yaml"

    export_best_config(
        {
            "parameters": {"max_num_seqs": 8},
            "metrics": {"throughput_requests_per_sec": 2.0},
            "trial_number": 4,
            "state": "COMPLETE",
        },
        output,
    )

    exported = yaml.safe_load(output.read_text())
    assert exported["vllm_params"] == {"max_num_seqs": 8}
    assert exported["performance_metrics"]["throughput_requests_per_sec"] == 2.0
    assert exported["candidate_info"]["status"] == "COMPLETE"
    assert "model" not in exported


def test_list_studies_shows_manifest_best_status_and_legacy_label(tmp_path: Path) -> None:
    _write_experiment(tmp_path)
    legacy = tmp_path / "old" / "configs"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(json.dumps({"best_trial": {}}))

    result = runner.invoke(app, ["list-studies", "--results-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Found 2 studies" in result.output
    assert "Status: COMPLETE (summary.json present)" in result.output
    assert "Manifest: model=fake/model" in result.output
    assert (
        "Best: trial=default-0, status=COMPLETE, goodput=3.500 req/s, validated=yes"
        in result.output
    )
    assert "old [legacy layout]" in result.output
    assert "not interpreted as SLOTune" in result.output


def test_report_rejects_legacy_layout_instead_of_mixing_schemas(tmp_path: Path) -> None:
    legacy = tmp_path / "old" / "configs"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(json.dumps({"best_trial": {}}))

    result = runner.invoke(
        app,
        ["report", "-n", "old", "--results-root", str(tmp_path), "-f", "json"],
    )

    assert result.exit_code == 1
    assert "Legacy study layout detected" in result.output


def test_tune_help_exposes_manifest_validated_resume() -> None:
    result = runner.invoke(app, ["tune", "--help"])

    assert result.exit_code == 0, result.output
    help_output = strip_ansi(result.output)
    assert "--resume" in help_output
    assert "--allow-dirty-source" in help_output

    tune_command = get_command(app).commands["tune"]
    options = {parameter.name: parameter for parameter in tune_command.params}
    resume_option = options["resume"]
    dirty_source_option = options["allow_dirty_source"]
    assert "--resume" in resume_option.opts
    assert "immutable" in resume_option.help
    assert "manifest" in resume_option.help
    assert "--allow-dirty-source" in dirty_source_option.opts
    assert "formal runs" in dirty_source_option.help
    assert "require clean Git" in dirty_source_option.help
