"""CLI interface for vLLM tuner using Typer."""

import asyncio
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import typer

from vllm_tuner.config.validation import (
    TunerSettings,
    load_yaml_config,
    validate_study_name,
)
from vllm_tuner.experiment.artifacts import (
    EXPERIMENT_INTEGRITY_FILE,
    SUMMARY_COMPACT_FILE,
    ArtifactStore,
)
from vllm_tuner.experiment.manifest import git_state, source_tree_sha256

logging.getLogger("httpx").setLevel(logging.WARNING)

app = typer.Typer(
    name="vllm-tuner",
    help="Auto-tuner for vLLM to maximize throughput and minimize latency",
    add_completion=False,
)

settings = TunerSettings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = "/root/autodl-tmp/slotune-results"
AGGREGATE_TABLES = {
    "trials": "trials.parquet",
    "repetitions": "repeated-results.parquet",
    "holdout": "holdout-results.parquet",
    "candidate_validation": "candidate-validation.parquet",
    "capacity_sweep": "capacity-sweep.parquet",
    "capacity_sweep_summary": "capacity-sweep-summary.parquet",
}


@app.command()
def tune(
    config: str = typer.Option(
        "config/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    study_name: str = typer.Option(
        None,
        "--study-name",
        "-n",
        help="Name of the tuning study (required)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Override model name",
    ),
    gpu_count: Optional[int] = typer.Option(
        None,
        "--gpu-count",
        "-g",
        help="Override number of GPUs",
    ),
    with_progress: bool = typer.Option(
        False,
        "--with-progress",
        help="Enable progress bar (disabled by default)",
    ),
    generate_baseline: bool = typer.Option(
        False,
        "--baseline/--no-baseline",
        help="Deprecated; the equal-budget default method is always the baseline",
    ),
    results_root: str = typer.Option(
        DEFAULT_RESULTS_ROOT,
        "--results-root",
        help="Data-disk root for immutable experiment artifacts",
    ),
    trace: Optional[str] = typer.Option(
        None,
        "--trace",
        help="Optional fixed search JSONL trace",
    ),
    holdout_trace: Optional[str] = typer.Option(
        None,
        "--holdout-trace",
        help="Optional fixed holdout JSONL trace",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume only after validating the immutable manifest and cached trial evidence",
    ),
    allow_dirty_source: bool = typer.Option(
        False,
        "--allow-dirty-source",
        help="Development only: permit an uncommitted source tree (formal runs require clean Git)",
    ),
):
    """Run a complete SLO-goodput search, repeat, holdout, and report experiment."""
    if study_name is None:
        study_name = "slotune"

    study_name = validate_study_name(study_name)
    logger.info("Starting SLOTune experiment: %s", study_name)

    try:
        config_obj = load_yaml_config(config)
        if model:
            config_obj.model = model
        if resume:
            config_obj.study.resume = True
        if gpu_count is not None and gpu_count != 1:
            raise ValueError("SLOTune core supports exactly one GPU")
        if generate_baseline:
            typer.echo(
                "The separate legacy baseline is disabled; method=default uses the same trial "
                "budget as random and TPE."
            )
        if with_progress:
            typer.echo("Progress is recorded in per-trial status.json artifacts.")

        from vllm_tuner.experiment.runner import SLOTuneExperimentRunner

        runner = SLOTuneExperimentRunner(
            config_obj,
            study_name,
            results_root=results_root,
            repository=Path.cwd(),
            trace_path=trace,
            holdout_trace_path=holdout_trace,
            require_clean_source=not allow_dirty_source,
        )
        summary = asyncio.run(runner.run())
        best = summary.get("best")
        typer.echo(f"\nSLOTune experiment '{study_name}' completed.")
        if best:
            typer.echo(
                "Best validated SLO goodput: "
                f"{_display(best.get('goodput_requests_per_sec'))} requests/s"
            )
            typer.echo(f"Parameters: {best.get('parameters', {})}")
        else:
            typer.echo(
                "No strictly validated candidate was found; inspect search_best, "
                "candidate_validation, and structured failure artifacts."
            )
        typer.echo(f"Artifacts: {runner.artifacts.root}")
        typer.echo(f"Report: {summary['report']['html']}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Tune command failed: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _experiment_directory(results_root: str, study_name: str) -> Path:
    """Resolve one sanitized experiment under the selected results root."""
    return Path(results_root).expanduser().resolve() / validate_study_name(study_name)


def _experiment_store(experiment_dir: Path) -> ArtifactStore:
    return ArtifactStore(experiment_dir.parent, experiment_dir.name)


def _validate_sealed_experiment(store: ArtifactStore) -> bool:
    if not (store.root / EXPERIMENT_INTEGRITY_FILE).is_file():
        return False
    store.validate_experiment_integrity()
    return True


def _inside_experiment(path: Path, experiment_dir: Path) -> bool:
    try:
        path.resolve().relative_to(experiment_dir.resolve())
    except ValueError:
        return False
    return True


def _require_safe_experiment_output(
    path: Path,
    experiment_dir: Path,
    *,
    kind: str,
) -> None:
    if not _inside_experiment(path, experiment_dir):
        return
    relative = path.resolve().relative_to(experiment_dir.resolve())
    if kind == "report":
        reserved = {
            "report.html",
            "report.md",
            "plot-manifest.json",
            "comparison-table.md",
            "capacity-curve.html",
            "capacity-curve.png",
            "pareto.html",
            "pareto.png",
            "search-trajectory.html",
            "search-trajectory.png",
            "telemetry-timeline.html",
            "telemetry-timeline.png",
            "scheduler-negative-results.md",
        }
        if len(relative.parts) < 2 or relative.parts[0] != "report" or relative.name in reserved:
            raise ValueError(
                "Report output inside an experiment must be a new export under report/; "
                "identity, raw, trial, and canonical report artifacts cannot be overwritten"
            )
        return
    if kind == "best" and (
        relative.parent != Path(".") or relative.name not in {"best.yaml", "best.json"}
    ):
        raise ValueError("Best export inside an experiment must be exactly best.yaml or best.json")


def _reseal_cli_write(store: ArtifactStore, *, kind: str, was_sealed: bool) -> None:
    if not was_sealed:
        return
    store.seal_experiment_artifacts(attestation=_attestation_identity(kind))
    store.validate_experiment_integrity()


def _attestation_identity(kind: str) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    commit, dirty, _ = git_state(repository)
    return {
        "kind": kind,
        "attestation_source_commit": commit,
        "attestation_source_tree_sha256": source_tree_sha256(repository),
        "attestation_dirty_worktree": dirty,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object with an artifact-specific error message."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_slotune_summary(experiment_dir: Path) -> dict[str, Any]:
    """Load only the current immutable experiment layout."""
    compact_path = experiment_dir / SUMMARY_COMPACT_FILE
    summary_path = compact_path if compact_path.is_file() else experiment_dir / "summary.json"
    if not summary_path.exists():
        legacy_path = experiment_dir / "configs" / "summary.json"
        if legacy_path.exists():
            raise ValueError(
                f"Legacy study layout detected at {legacy_path}; this command expects "
                "a SLOTune experiment summary at <results-root>/<experiment>/summary.json"
            )
        raise FileNotFoundError(f"SLOTune summary not found: {summary_path}")
    summary = _read_json_object(summary_path)
    if "experiment_id" not in summary or "manifest" not in summary:
        raise ValueError(
            f"{summary_path} is not a SLOTune experiment summary; legacy and current "
            "study formats are not mixed"
        )
    if not isinstance(summary["manifest"], dict):
        raise ValueError(f"{summary_path} contains a non-object manifest")
    return summary


def _read_aggregate_tables(experiment_dir: Path) -> dict[str, dict[str, Any]]:
    """Read aggregate Parquet tables while representing missing data explicitly."""
    import pandas as pd  # type: ignore[import-untyped]

    tables: dict[str, dict[str, Any]] = {}
    for label, filename in AGGREGATE_TABLES.items():
        path = experiment_dir / "aggregate" / filename
        record: dict[str, Any] = {
            "path": path.relative_to(experiment_dir).as_posix(),
            "available": False,
            "row_count": None,
            "records": [],
        }
        if not path.exists():
            record["unavailable_reason"] = "artifact does not exist"
        else:
            try:
                frame = pd.read_parquet(path)
                records = json.loads(frame.to_json(orient="records"))
            except Exception as error:
                record["unavailable_reason"] = f"{type(error).__name__}: {error}"
            else:
                record.update(
                    {
                        "available": True,
                        "row_count": len(frame),
                        "records": records,
                        "unavailable_reason": None,
                    }
                )
        tables[label] = record
    return tables


def _report_artifacts(
    experiment_dir: Path, summary: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Resolve expected report files without trusting stale absolute paths alone."""
    filenames = {
        "html": "report.html",
        "markdown": "report.md",
        "plot_manifest": "plot-manifest.json",
    }
    report_value = summary.get("report")
    report_mapping = report_value if isinstance(report_value, Mapping) else {}
    artifacts: dict[str, dict[str, Any]] = {}
    for label, filename in filenames.items():
        expected = experiment_dir / "report" / filename
        candidates = [expected]
        configured = report_mapping.get(label)
        if isinstance(configured, str):
            configured_path = Path(configured).expanduser()
            candidates.append(
                configured_path
                if configured_path.is_absolute()
                else experiment_dir / configured_path
            )
        source = next((candidate for candidate in candidates if candidate.is_file()), expected)
        artifacts[label] = {
            "path": str(source),
            "available": source.is_file(),
            "unavailable_reason": None if source.is_file() else "artifact does not exist",
        }
    return artifacts


def _validated_best(
    summary: Mapping[str, Any],
) -> tuple[Optional[dict[str, Any]], dict[str, list[Mapping[str, Any]]], str]:
    """Trust only the runner's strict repeat-and-holdout validation verdict."""
    empty_evidence: dict[str, list[Mapping[str, Any]]] = {"repeats": [], "holdouts": []}
    raw_best = summary.get("best")
    if not isinstance(raw_best, dict) or not raw_best:
        return None, empty_evidence, "summary.best is empty; no feasible candidate was selected"
    parameters = raw_best.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        return None, empty_evidence, "summary.best.parameters is empty"
    if raw_best.get("status") not in {None, "COMPLETE"} or raw_best.get("feasible") is False:
        return None, empty_evidence, "summary.best is not a COMPLETE feasible candidate"
    if raw_best.get("validated") is not True:
        return None, empty_evidence, "summary.best.validated is not true"
    if raw_best.get("metric_provenance") != "median_of_complete_feasible_repeats":
        return None, empty_evidence, "summary.best metric provenance is not repeat aggregation"
    if any(
        field in raw_best
        for field in ("trial_id", "trial_number", "search_goodput_requests_per_sec")
    ):
        return None, empty_evidence, "summary.best mixes a search-trial identity or metric"

    method = raw_best.get("method")
    candidate = raw_best.get("candidate")
    search_observation = raw_best.get("search_observation")
    if (
        not isinstance(method, str)
        or not isinstance(candidate, str)
        or not isinstance(search_observation, Mapping)
        or not isinstance(search_observation.get("trial_number"), int)
    ):
        return None, empty_evidence, "summary.best candidate/search provenance is incomplete"
    source_number = search_observation["trial_number"]
    if (
        search_observation.get("method") != method
        or search_observation.get("parameters") != parameters
    ):
        return (
            None,
            empty_evidence,
            "summary.best search_observation does not identify its candidate",
        )

    expected = raw_best.get("repeat_required")
    if not isinstance(expected, int) or expected < 1:
        return None, empty_evidence, "summary.best.repeat_required is invalid"
    if raw_best.get("holdout_required") is not True:
        return None, empty_evidence, "summary.best has no required holdout validation"

    def matches(row: Any, *, holdout: bool) -> bool:
        return (
            isinstance(row, Mapping)
            and row.get("parameters") == parameters
            and row.get("method") == method
            and row.get("repeat_of") == source_number
            and row.get("holdout") is holdout
            and row.get("status") == "COMPLETE"
            and row.get("feasible") is True
        )

    raw_repetitions = summary.get("repetitions")
    repetitions = raw_repetitions if isinstance(raw_repetitions, list) else []
    matching_repeats = [row for row in repetitions if matches(row, holdout=False)]

    raw_holdout = summary.get("holdout")
    holdout_rows = raw_holdout if isinstance(raw_holdout, list) else []
    matching_holdouts = [row for row in holdout_rows if matches(row, holdout=True)]
    evidence = {"repeats": matching_repeats, "holdouts": matching_holdouts}
    if len(matching_repeats) != expected or len(matching_holdouts) != expected:
        return None, evidence, "summary does not contain every exact repeat and holdout row"
    if (
        raw_best.get("repeat_complete_feasible") != expected
        or raw_best.get("holdout_complete_feasible") != expected
    ):
        return None, evidence, "summary.best repeat/holdout counts are inconsistent"

    repeat_metrics = raw_best.get("repeat_metrics")
    holdout_metrics = raw_best.get("holdout_metrics")
    if not isinstance(repeat_metrics, Mapping) or not isinstance(holdout_metrics, Mapping):
        return None, evidence, "summary.best repeat/holdout aggregates are missing"
    for label, metrics in (("repeat", repeat_metrics), ("holdout", holdout_metrics)):
        goodput = metrics.get("goodput_requests_per_sec")
        if not isinstance(goodput, Mapping) or goodput.get("count") != expected:
            return None, evidence, f"summary.best {label} goodput count is inconsistent"
    for metric, aggregate in repeat_metrics.items():
        if isinstance(aggregate, Mapping) and raw_best.get(metric) != aggregate.get("median"):
            return None, evidence, f"summary.best canonical {metric} is not the repeat median"

    validation_rows_value = summary.get("candidate_validation")
    validation_rows = validation_rows_value if isinstance(validation_rows_value, list) else []
    matching_validation = [
        row
        for row in validation_rows
        if isinstance(row, Mapping)
        and row.get("candidate") == candidate
        and row.get("method") == method
        and row.get("parameters") == parameters
        and row.get("validated") is True
    ]
    if len(matching_validation) != 1:
        return None, evidence, "summary candidate-validation verdict is missing or ambiguous"
    return (
        raw_best,
        evidence,
        "validated by the runner's strict repeat-and-all-holdout policy",
    )


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}" if math.isfinite(value) else "unavailable"
    return str(value)


def _metric_summary(aggregates: Mapping[str, Any], metric: str) -> str:
    record = aggregates.get(metric)
    if not isinstance(record, Mapping):
        return "unavailable"
    return (
        f"{_display(record.get('median'))} "
        f"({_display(record.get('min'))}–{_display(record.get('max'))}; "
        f"n={_display(record.get('count'))})"
    )


def _generate_slotune_markdown(
    summary: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render a concise current-schema summary without legacy metric aliases."""
    manifest_value = summary.get("manifest")
    manifest = manifest_value if isinstance(manifest_value, Mapping) else {}
    best_value = summary.get("best")
    best = best_value if isinstance(best_value, Mapping) else {}
    validated, validation_evidence, validation_reason = _validated_best(summary)
    repeat_metrics_value = best.get("repeat_metrics")
    repeat_metrics = repeat_metrics_value if isinstance(repeat_metrics_value, Mapping) else {}
    holdout_metrics_value = best.get("holdout_metrics")
    holdout_metrics = holdout_metrics_value if isinstance(holdout_metrics_value, Mapping) else {}
    search_value = best.get("search_observation")
    search_observation = search_value if isinstance(search_value, Mapping) else {}
    parameters = best.get("parameters")
    parameter_rows = (
        "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(parameters.items()))
        if isinstance(parameters, Mapping) and parameters
        else "- unavailable"
    )
    aggregate_rows = "\n".join(
        f"| {label} | {_display(record.get('row_count'))} | "
        f"{'available' if record.get('available') else 'unavailable'} |"
        for label, record in aggregates.items()
    )
    artifact_rows = "\n".join(
        f"- {label}: `{record.get('path')}` "
        f"({'available' if record.get('available') else 'unavailable'})"
        for label, record in artifacts.items()
    )
    warnings_value = manifest.get("artifact_warnings")
    warnings = warnings_value if isinstance(warnings_value, list) else []
    warnings_text = "\n".join(f"- {warning}" for warning in warnings) or "- None recorded."
    return f"""# SLOTune experiment: {summary.get('experiment_id', 'unknown')}

## Identity

- Model: `{manifest.get('model', 'unavailable')}`
- Source commit: `{manifest.get('source_commit') or 'unavailable'}`
- Trace SHA-256: `{manifest.get('trace_sha256', 'unavailable')}`
- Holdout trace SHA-256: `{manifest.get('holdout_trace_sha256') or 'unavailable'}`

## Best validated candidate

- Candidate: `{best.get('candidate', 'unavailable')}`
- Source search trial: `{search_observation.get('trial_id', 'unavailable')}`
- Status: `{best.get('status', 'unavailable')}`
- Feasible: `{_display(best.get('feasible'))}`
- Metric provenance: `{best.get('metric_provenance', 'unavailable')}`
- SLO goodput, repeat median (range): `{_metric_summary(repeat_metrics, 'goodput_requests_per_sec')}` requests/s
- Achieved throughput, repeat median (range): `{_metric_summary(repeat_metrics, 'achieved_requests_per_sec')}` requests/s
- p99 TTFT, repeat median (range): `{_metric_summary(repeat_metrics, 'p99_ttft_ms')}` ms
- Holdout SLO goodput, median (range): `{_metric_summary(holdout_metrics, 'goodput_requests_per_sec')}` requests/s
- Search SLO goodput (selection observation only): `{_display(search_observation.get('goodput_requests_per_sec'))}` requests/s
- Holdout validation: `{'passed' if validated is not None else 'failed'}` ({validation_reason}; matching repeats: {len(validation_evidence['repeats'])}; matching holdouts: {len(validation_evidence['holdouts'])})

### Parameters

{parameter_rows}

## Aggregate evidence

| Table | Rows | Availability |
|---|---:|---|
{aggregate_rows}

## Existing report artifacts

{artifact_rows}

## Artifact warnings

{warnings_text}
"""


@app.command()
def attest(
    study_name: str = typer.Option(
        ...,
        "--study-name",
        "-n",
        help="Completed SLOTune experiment to attest",
    ),
    results_root: str = typer.Option(
        DEFAULT_RESULTS_ROOT,
        "--results-root",
        help="Root containing immutable SLOTune experiment directories",
    ),
    reseal: bool = typer.Option(
        False,
        "--reseal",
        help="Explicitly authorize regeneration after validating an existing root seal",
    ),
):
    """Compact and attest a completed experiment without changing sealed trial/raw evidence."""
    try:
        experiment_dir = _experiment_directory(results_root, study_name)
        _load_slotune_summary(experiment_dir)
        store = _experiment_store(experiment_dir)
        result = store.attest_experiment_artifacts(
            attestation=_attestation_identity("cli-post-run-attestation"),
            reseal=reseal,
        )
        if result["already_sealed"]:
            typer.echo(
                f"Experiment root seal is valid and unchanged: "
                f"{experiment_dir / EXPERIMENT_INTEGRITY_FILE}"
            )
        else:
            audit = result["audit"]
            typer.echo(
                f"Experiment attested: {experiment_dir / EXPERIMENT_INTEGRITY_FILE} "
                f"({audit['trial_semantic_validated']}/{audit['trial_count']} trials validated)"
            )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Attest command failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    study_name: str = typer.Option(
        ...,
        "--study-name",
        "-n",
        help="Name of the study to report",
    ),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Report format (html, json, markdown)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path; HTML defaults to the existing immutable report artifact",
    ),
    results_root: str = typer.Option(
        DEFAULT_RESULTS_ROOT,
        "--results-root",
        help="Root containing immutable SLOTune experiment directories",
    ),
):
    """Reuse or export a report from one immutable SLOTune experiment."""
    try:
        selected_format = format.lower()
        if selected_format not in {"html", "json", "markdown"}:
            raise ValueError(f"Unsupported format '{format}'")
        experiment_dir = _experiment_directory(results_root, study_name)
        store = _experiment_store(experiment_dir)
        was_sealed = _validate_sealed_experiment(store)
        summary = _load_slotune_summary(experiment_dir)
        aggregates = _read_aggregate_tables(experiment_dir)
        artifacts = _report_artifacts(experiment_dir, summary)

        if selected_format == "html":
            html_record = artifacts["html"]
            if not html_record["available"]:
                raise FileNotFoundError(
                    f"Static SLOTune HTML report not found: {html_record['path']}"
                )
            source = Path(str(html_record["path"])).resolve()
            if output is None:
                typer.echo(f"HTML report reused: {source}")
                return
            destination = Path(output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination != source:
                _require_safe_experiment_output(destination, experiment_dir, kind="report")
                shutil.copy2(source, destination)
                if _inside_experiment(destination, experiment_dir):
                    _reseal_cli_write(
                        store,
                        kind="cli-report-html-copy",
                        was_sealed=was_sealed,
                    )
                typer.echo(f"HTML report copied to: {destination}")
            else:
                typer.echo(f"HTML report reused: {source}")
            return

        suffix = "json" if selected_format == "json" else "md"
        destination = (
            Path(output).expanduser().resolve()
            if output is not None
            else experiment_dir / "report" / f"report-export.{suffix}"
        )
        _require_safe_experiment_output(destination, experiment_dir, kind="report")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if selected_format == "json":
            payload = {
                "schema": "slotune-experiment-report-v1",
                "summary": summary,
                "aggregates": aggregates,
                "report_artifacts": artifacts,
            }
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if _inside_experiment(destination, experiment_dir):
                _reseal_cli_write(
                    store,
                    kind="cli-report-json-export",
                    was_sealed=was_sealed,
                )
            typer.echo(f"JSON report exported: {destination}")
        else:
            destination.write_text(
                _generate_slotune_markdown(summary, aggregates, artifacts),
                encoding="utf-8",
            )
            if _inside_experiment(destination, experiment_dir):
                _reseal_cli_write(
                    store,
                    kind="cli-report-markdown-export",
                    was_sealed=was_sealed,
                )
            typer.echo(f"Markdown report exported: {destination}")
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Report command failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command()
def export(
    study_name: str = typer.Option(
        ...,
        "--study-name",
        "-n",
        help="Name of the study to export",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <results-root>/<experiment>/best.<format>)",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Export format (yaml, json)",
    ),
    results_root: str = typer.Option(
        DEFAULT_RESULTS_ROOT,
        "--results-root",
        help="Root containing immutable SLOTune experiment directories",
    ),
):
    """Export a best configuration only after successful holdout validation."""
    try:
        selected_format = format.lower()
        if selected_format not in {"yaml", "json"}:
            raise ValueError(f"Unsupported format '{format}'")
        experiment_dir = _experiment_directory(results_root, study_name)
        store = _experiment_store(experiment_dir)
        was_sealed = _validate_sealed_experiment(store)
        summary = _load_slotune_summary(experiment_dir)
        best, validation_evidence, validation_reason = _validated_best(summary)
        if best is None:
            raise ValueError(f"Best configuration is not exportable: {validation_reason}")
        if output is None:
            output_path = experiment_dir / f"best.{selected_format}"
        else:
            output_path = Path(output).expanduser().resolve()
        _require_safe_experiment_output(output_path, experiment_dir, kind="best")

        from vllm_tuner.reporting.export import export_best_config

        export_best_config(
            best,
            output_path,
            selected_format,
            experiment_id=str(summary["experiment_id"]),
            manifest=(summary["manifest"] if isinstance(summary.get("manifest"), Mapping) else {}),
            validation={
                "validated": True,
                "method": "strict repeat-and-all-holdout",
                "matching_repeat_trial_ids": [
                    row.get("trial_id") for row in validation_evidence["repeats"]
                ],
                "matching_holdout_trial_ids": [
                    row.get("trial_id") for row in validation_evidence["holdouts"]
                ],
            },
        )
        if _inside_experiment(output_path, experiment_dir):
            _reseal_cli_write(store, kind="cli-best-export", was_sealed=was_sealed)
        typer.echo(f"Validated best configuration exported to: {output_path}")
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Export command failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command("longctx-m0")
def longctx_m0(
    config: str = typer.Option(
        "experiments/long_context/v5/m0-production-default.yaml",
        "--config",
        "-c",
        help="Strict long-context v5 M0 YAML configuration",
    ),
    experiment_id: str = typer.Option(
        "longctx-v5-m0-qwen25-7b-production-default-001",
        "--experiment-id",
        "-n",
        help="Fresh v5 M0 artifact directory name",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Validate identity and replay a sealed or cached complete M0 trial",
    ),
    allow_dirty_source: bool = typer.Option(
        False,
        "--allow-dirty-source",
        help="Development smoke only; a dirty run cannot pass M0 acceptance",
    ),
):
    """Run one isolated 100+ request v5 production-default canary."""
    try:
        from vllm_tuner.longctx.m0_config import load_longctx_m0_config
        from vllm_tuner.longctx.m0_runner import LongContextM0Runner

        config_obj = load_longctx_m0_config(config)
        runner = LongContextM0Runner(
            config_obj,
            experiment_id,
            repository=Path(__file__).resolve().parents[3],
            resume=resume,
            require_clean_source=not allow_dirty_source,
        )
        summary = asyncio.run(runner.run())
        acceptance = summary.get("acceptance", {})
        is_smoke = config_obj.evidence_role == "smoke"
        status_key = "execution_passed" if is_smoke else "passed"
        passed = isinstance(acceptance, Mapping) and acceptance.get(status_key) is True
        status_label = "Smoke execution status" if is_smoke else "M0 status"
        typer.echo(f"{status_label}: {'PASS' if passed else 'FAIL'}")
        if is_smoke:
            typer.echo("Formal M0 qualification: not applicable to smoke artifacts")
        typer.echo(f"Trial: {summary.get('trial_id', 'unavailable')}")
        typer.echo(f"Artifacts: {runner.artifacts.root}")
        if summary.get("resume_replayed") is True:
            typer.echo("Resume: sealed artifact replayed; no server was started")
        elif isinstance(summary.get("resume"), Mapping):
            typer.echo(
                "Resume: cached trial replayed"
                if summary["resume"].get("trial_replayed") is True
                else "Resume: new trial executed"
            )
        if not passed:
            typer.echo(
                (
                    "Smoke execution failed; inspect m0-summary.json."
                    if is_smoke
                    else "M0 acceptance failed; inspect m0-summary.json and do not enter M1."
                ),
                err=True,
            )
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Long-context M0 command failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command("longctx-m1-init")
def longctx_m1_init(
    config: str = typer.Option(
        "experiments/long_context/v5/m1-initialization.yaml",
        "--config",
        "-c",
        help="Strict long-context v5 M1 initialization matrix",
    ),
    experiment_id: str = typer.Option(
        "longctx-v5-m1-planner-init-001",
        "--experiment-id",
        "-n",
        help="Fresh M1 initialization-validation artifact directory",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Replay only checksum-valid completed probes or a sealed root",
    ),
):
    """Run M1 KV Planner calibration and held-out initialization validation."""
    try:
        from vllm_tuner.longctx.m1_config import load_longctx_m1_config
        from vllm_tuner.longctx.m1_runner import LongContextM1Runner

        config_obj = load_longctx_m1_config(config)
        runner = LongContextM1Runner(
            config_obj,
            experiment_id,
            repository=Path(__file__).resolve().parents[3],
            resume=resume,
        )
        summary = asyncio.run(runner.run())
        primary_passed = summary.get("primary_error_passed") is True
        extrapolation_passed = summary.get("extrapolation_error_passed") is True
        passed = summary.get("initialization_validation_passed") is True
        typer.echo(f"M1 initialization validation: {'PASS' if passed else 'FAIL'}")
        typer.echo(f"In-profile held-out: {'PASS' if primary_passed else 'FAIL'}")
        typer.echo(f"Context extrapolation: {'PASS' if extrapolation_passed else 'FAIL'}")
        typer.echo(f"Validation points: {len(summary.get('validations', []))}")
        typer.echo(f"Artifacts: {runner.store.root}")
        if summary.get("resume_replayed") is True:
            typer.echo("Resume: sealed artifact replayed; no server was started")
        if not passed:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Long-context M1 init command failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command("longctx-m1-capacity")
def longctx_m1_capacity(
    config: str = typer.Option(
        "experiments/long_context/v5/m1-capacity-smoke.yaml",
        "--config",
        "-c",
        help="Strict long-context v5 M1 capacity smoke, pilot, or formal matrix",
    ),
    experiment_id: str = typer.Option(
        "longctx-v5-m1-capacity-smoke-001",
        "--experiment-id",
        "-n",
        help="Fresh M1 capacity-sweep artifact directory name",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Validate identity and replay only checksum-valid completed capacity points",
    ),
):
    """Run an isolated v5 M1 capacity sweep without changing upstream defaults."""
    try:
        from vllm_tuner.longctx.m1_capacity_config import (
            load_longctx_m1_capacity_config,
        )
        from vllm_tuner.longctx.m1_capacity_runner import LongContextM1CapacityRunner

        config_obj = load_longctx_m1_capacity_config(config)
        runner = LongContextM1CapacityRunner(
            config_obj,
            experiment_id,
            repository=Path(__file__).resolve().parents[3],
            resume=resume,
        )
        summary = asyncio.run(runner.run())
        execution = summary.get("execution", {})
        acceptance = summary.get("acceptance", {})
        artifacts = summary.get("artifacts", {})
        execution_passed = isinstance(execution, Mapping) and execution.get("passed") is True
        acceptance_passed = isinstance(acceptance, Mapping) and acceptance.get("passed") is True

        typer.echo(f"Execution: {'PASS' if execution_passed else 'FAIL'}")
        if isinstance(execution, Mapping):
            typer.echo(
                "Capacity jobs: "
                f"{execution.get('completed_jobs', 'unavailable')}/"
                f"{execution.get('planned_jobs', 'unavailable')} completed; "
                f"{execution.get('failed_jobs', 'unavailable')} failed"
            )
        if config_obj.evidence_role == "formal":
            typer.echo(f"M1 capacity acceptance: {'PASS' if acceptance_passed else 'FAIL'}")
            command_passed = acceptance_passed
        else:
            typer.echo(
                "M1 capacity acceptance: NOT ELIGIBLE "
                f"({config_obj.evidence_role} evidence is execution-only)"
            )
            command_passed = execution_passed
        artifact_root = (
            artifacts.get("root")
            if isinstance(artifacts, Mapping)
            else config_obj.artifacts.root / experiment_id
        )
        typer.echo(f"Artifacts: {artifact_root}")
        resume_summary = summary.get("resume")
        if isinstance(resume_summary, Mapping):
            typer.echo(
                "Resume: "
                f"{resume_summary.get('replayed_trials', 0)} replayed, "
                f"{resume_summary.get('new_attempts', 0)} new attempts"
            )
        if not command_passed:
            message = (
                "Formal M1 capacity acceptance failed; inspect the summary and do not enter M2."
                if config_obj.evidence_role == "formal"
                else "Capacity smoke/pilot execution failed; inspect the summary before continuing."
            )
            typer.echo(message, err=True)
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Long-context M1 capacity command failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command("longctx-m1-capacity-boundaries")
def longctx_m1_capacity_boundaries(
    artifact_root: str = typer.Option(
        "/root/autodl-tmp/longctx-v5-artifacts",
        "--artifact-root",
        help="Root containing the sealed source and derived M1 artifacts",
    ),
    source_experiment_id: str = typer.Option(
        "longctx-v5-m1-capacity-formal-001",
        "--source-experiment-id",
        help="Immutable sealed v1 formal capacity experiment",
    ),
    experiment_id: str = typer.Option(
        "longctx-v5-m1-capacity-formal-001-boundaries-v2",
        "--experiment-id",
        "-n",
        help="Fresh zero-GPU M1 v2 boundary artifact directory name",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Validate and replay an existing sealed v2 boundary artifact",
    ),
):
    """Derive separate SLO service and joint saturation boundaries from sealed M1 data."""
    try:
        from vllm_tuner.longctx.m1_capacity_reanalysis import M1CapacityBoundaryRunner

        runner = M1CapacityBoundaryRunner(
            artifact_root,
            source_experiment_id,
            experiment_id,
            repository=Path(__file__).resolve().parents[3],
            resume=resume,
        )
        summary = runner.run()
        source_acceptance = summary.get("source_v1_acceptance", {})
        acceptance = summary.get("acceptance", {})
        artifacts = summary.get("artifacts", {})
        source_passed = (
            isinstance(source_acceptance, Mapping) and source_acceptance.get("passed") is True
        )
        passed = isinstance(acceptance, Mapping) and acceptance.get("passed") is True
        typer.echo(f"Source v1 acceptance preserved: {'PASS' if source_passed else 'FAIL'}")
        typer.echo(f"M1 v2 boundary acceptance: {'PASS' if passed else 'FAIL'}")
        typer.echo(f"GPU runs executed: {summary.get('gpu_runs_executed', 'unavailable')}")
        artifact_path = (
            artifacts.get("root", runner.store.root)
            if isinstance(artifacts, Mapping)
            else runner.store.root
        )
        typer.echo(f"Artifacts: {artifact_path}")
        typer.echo("M2 was not started; review the independent M1 milestone result first.")
        if not passed:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    except Exception as error:
        logger.error("Long-context M1 boundary analysis failed: %s", error, exc_info=True)
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


def _longctx_status_value(value: object) -> str:
    """Format nested status values predictably for shell operators."""
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


@app.command("longctx-m1-capacity-status")
def longctx_m1_capacity_status(
    artifact_root: str = typer.Option(
        "/root/autodl-tmp/longctx-v5-artifacts",
        "--artifact-root",
        help="Root containing running or sealed long-context v5 capacity artifacts",
    ),
    experiment_id: str = typer.Option(
        "longctx-v5-m1-capacity-smoke-001",
        "--experiment-id",
        "-n",
        help="Existing M1 capacity-sweep artifact directory name",
    ),
):
    """Display unattended-run state and exact continuation paths for one M1 sweep."""
    try:
        from vllm_tuner.longctx.m1_capacity_runner import load_m1_capacity_status

        root = Path(artifact_root).expanduser().resolve()
        status = load_m1_capacity_status(root, experiment_id)
        for label, key in (
            ("State", "state"),
            ("PID", "pid"),
            ("GPU", "gpu"),
            ("Log", "log"),
            ("Result", "result"),
            ("ETA", "eta"),
            ("Resume", "resume"),
            ("Sealed", "sealed"),
            ("Acceptance", "acceptance"),
        ):
            typer.echo(f"{label}: {_longctx_status_value(status.get(key))}")
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command("longctx-m0-status")
def longctx_m0_status(
    artifact_root: str = typer.Option(
        "/root/autodl-tmp/longctx-v5-artifacts",
        "--artifact-root",
        help="Root containing sealed long-context v5 artifacts",
    ),
    experiment_id: str = typer.Option(
        "longctx-v5-m0-qwen25-7b-production-default-001",
        "--experiment-id",
        "-n",
        help="Existing sealed v5 M0 artifact directory name",
    ),
):
    """Validate and display one sealed v5 M0 artifact root."""
    try:
        from vllm_tuner.longctx.m0_runner import load_m0_status

        root = Path(artifact_root).expanduser().resolve()
        summary = load_m0_status(root, experiment_id)
        acceptance = summary.get("acceptance", {})
        is_smoke = summary.get("evidence_role") == "smoke"
        status_key = "execution_passed" if is_smoke else "passed"
        passed = isinstance(acceptance, Mapping) and acceptance.get(status_key) is True
        status_label = "Smoke execution status" if is_smoke else "M0 status"
        typer.echo(f"{status_label}: {'PASS' if passed else 'FAIL'}")
        typer.echo(f"Trial: {summary.get('trial_id', 'unavailable')}")
        typer.echo(f"Artifacts: {root / experiment_id}")
        if not passed:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)


@app.command()
def list_studies(
    results_root: str = typer.Option(
        DEFAULT_RESULTS_ROOT,
        "--results-root",
        help="Root containing immutable SLOTune experiment directories",
    ),
):
    """List immutable SLOTune experiments and identify legacy layouts explicitly."""
    root = Path(results_root).expanduser().resolve()
    if not root.exists():
        typer.echo("No studies found")
        return
    studies = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and (
                (path / "summary.json").exists()
                or (path / "manifest.json").exists()
                or (path / "configs" / "summary.json").exists()
            )
        ),
        key=lambda path: path.name,
    )
    if not studies:
        typer.echo("No studies found")
        return
    typer.echo(f"Found {len(studies)} studies:")
    for study in studies:
        summary_path = study / "summary.json"
        manifest_path = study / "manifest.json"
        legacy_path = study / "configs" / "summary.json"
        if not summary_path.exists() and not manifest_path.exists() and legacy_path.exists():
            typer.echo(f"  - {study.name} [legacy layout]")
            typer.echo("    Status: legacy summary present; not interpreted as SLOTune")
            typer.echo(f"    Legacy summary: {legacy_path}")
            continue

        try:
            summary = _read_json_object(summary_path) if summary_path.exists() else {}
            if summary and ("experiment_id" not in summary or "manifest" not in summary):
                raise ValueError("summary.json is not the SLOTune schema")
            manifest = (
                _read_json_object(manifest_path)
                if manifest_path.exists()
                else summary.get("manifest", {})
            )
            if not isinstance(manifest, Mapping):
                raise ValueError("manifest is not a JSON object")
        except (OSError, ValueError) as error:
            typer.echo(f"  - {study.name}")
            typer.echo(f"    Status: ERROR ({error})")
            continue

        status = summary.get("status") if summary else None
        if status is None and summary_path.exists():
            status = "COMPLETE (summary.json present)"
        if status is None:
            status_files = sorted(
                (study / "trials").glob("*/status.json"),
                key=lambda path: path.stat().st_mtime,
            )
            if status_files:
                try:
                    status = _read_json_object(status_files[-1]).get("status")
                except (OSError, ValueError):
                    status = None
        status = status or "INCOMPLETE (manifest present, summary absent)"
        typer.echo(f"  - {study.name}")
        typer.echo(f"    Status: {status}")
        typer.echo(
            "    Manifest: model={model}, created={created}, commit={commit}".format(
                model=manifest.get("model", "unavailable"),
                created=manifest.get("created_at", "unavailable"),
                commit=manifest.get("source_commit") or "unavailable",
            )
        )
        best_value = summary.get("best") if summary else None
        if not isinstance(best_value, Mapping) or not best_value:
            typer.echo("    Best: unavailable")
            continue
        validated, validation_evidence, validation_reason = _validated_best(summary)
        typer.echo(
            "    Best: trial={trial}, status={status}, goodput={goodput} req/s, "
            "validated={validated}".format(
                trial=best_value.get("candidate", "unavailable"),
                status=best_value.get("status", "unavailable"),
                goodput=_display(best_value.get("goodput_requests_per_sec")),
                validated="yes" if validated is not None else "no",
            )
        )
        typer.echo(
            f"    Validation: {validation_reason}; "
            f"matching repeats={len(validation_evidence['repeats'])}; "
            f"matching holdouts={len(validation_evidence['holdouts'])}"
        )
        typer.echo(
            "    Parameters: "
            + json.dumps(best_value.get("parameters", {}), sort_keys=True, ensure_ascii=False)
        )


def main():
    app()


if __name__ == "__main__":
    main()
