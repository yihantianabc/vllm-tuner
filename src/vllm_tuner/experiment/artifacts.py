"""Atomic artifact layout and integrity checks for reproducible experiments."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel

from .manifest import git_state, sha256_file, source_tree_sha256
from .models import ExperimentSpec, TrialResult, trial_provenance, utc_now_iso

TRIAL_ARTIFACT_FILES = (
    "server-command.json",
    "params.json",
    "status.json",
    "request-results.jsonl",
    "benchmark-raw.json",
    "prometheus.jsonl",
    "nvml.jsonl",
    "server.log",
    "cleanup.json",
    "summary.json",
)

ARTIFACT_STATUS_FILE = "artifact-status.json"
ARTIFACT_INTEGRITY_FILE = "artifact-integrity.json"
EXPERIMENT_INTEGRITY_FILE = "experiment-integrity.json"
SUMMARY_COMPACT_FILE = "summary.compact-v1.json"
SEALED_TRIAL_FILES = (*TRIAL_ARTIFACT_FILES, ARTIFACT_STATUS_FILE)
REQUIRED_EXPERIMENT_INPUT_FILES = (
    "manifest.json",
    "experiment.yaml",
    "trace.jsonl",
    "trace.sha256",
    "holdout-trace.jsonl",
    "holdout-trace.sha256",
    "summary.json",
    "aggregate/scheduler-ablation.json",
    "report/report.md",
    "report/report.html",
    "report/plot-manifest.json",
)
GENERATED_EXPERIMENT_FILES = (
    SUMMARY_COMPACT_FILE,
    "lineage.json",
    "experiment-audit.json",
    "aggregate/scheduler-negative-results.json",
    "report/scheduler-negative-results.md",
)
REQUIRED_EXPERIMENT_FILES = (
    *REQUIRED_EXPERIMENT_INPUT_FILES,
    *GENERATED_EXPERIMENT_FILES,
)
CLEANUP_REQUIRED_TRUE_FIELDS = (
    "clean",
    "process_group_empty",
    "port_available",
    "gpu_clean",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class ArtifactStore:
    """Own the documented results directory and write complete files atomically."""

    def __init__(self, root: str | Path, experiment_id: str):
        self.root = Path(root).expanduser().resolve() / experiment_id
        self.trials_dir = self.root / "trials"
        self.aggregate_dir = self.root / "aggregate"
        self.report_dir = self.root / "report"
        self.environment_dir = self.root / "environment"

    def initialize(self, *, exist_ok: bool = False) -> Path:
        """Create an empty layout, refusing accidental experiment reuse by default."""
        if self.root.exists() and not exist_ok:
            raise FileExistsError(
                f"Experiment directory already exists: {self.root}; use explicit resume"
            )
        for directory in (
            self.trials_dir,
            self.aggregate_dir,
            self.report_dir,
            self.environment_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self.root

    @staticmethod
    def _atomic_text(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        """Write formatted JSON atomically."""
        payload = json.dumps(_json_value(value), indent=2, sort_keys=True, ensure_ascii=False)
        return self._atomic_text(self.root / relative, payload + "\n")

    def write_text(self, relative: str | Path, value: str) -> Path:
        """Write UTF-8 text atomically."""
        return self._atomic_text(self.root / relative, value)

    def write_yaml(self, relative: str | Path, value: Any) -> Path:
        """Write stable YAML atomically."""
        payload = yaml.safe_dump(_json_value(value), sort_keys=True, allow_unicode=True)
        return self._atomic_text(self.root / relative, payload)

    def write_jsonl(self, relative: str | Path, rows: Iterable[Any]) -> Path:
        """Write raw row-level records without losing failures or null values."""
        lines = [json.dumps(_json_value(row), sort_keys=True, ensure_ascii=False) for row in rows]
        return self._atomic_text(self.root / relative, "\n".join(lines) + ("\n" if lines else ""))

    def save_manifest(self, manifest: ExperimentSpec) -> Path:
        """Persist the experiment identity."""
        return self.write_json("manifest.json", manifest)

    def save_trace(self, trace_path: str | Path) -> tuple[Path, Path]:
        """Copy a fixed trace and save its checksum next to it."""
        return self._save_named_trace(trace_path, "trace.jsonl", "trace.sha256")

    def save_holdout_trace(self, trace_path: str | Path) -> tuple[Path, Path]:
        """Copy the frozen holdout trace and its checksum into the experiment."""
        return self._save_named_trace(trace_path, "holdout-trace.jsonl", "holdout-trace.sha256")

    def _save_named_trace(
        self, trace_path: str | Path, destination_name: str, checksum_name: str
    ) -> tuple[Path, Path]:
        source = Path(trace_path)
        destination = self.root / destination_name
        self._atomic_text(destination, source.read_text(encoding="utf-8"))
        checksum_path = self._atomic_text(
            self.root / checksum_name,
            f"{sha256_file(destination)}  {destination_name}\n",
        )
        return destination, checksum_path

    def trial_dir(self, trial_id: str) -> Path:
        """Return and create the isolated directory for one trial."""
        directory = self.trials_dir / str(trial_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_trial_result(self, result: TrialResult) -> Path:
        """Persist the final summary without overwriting state-machine history."""
        self.trial_dir(result.trial_id)
        return self.write_json(Path("trials") / result.trial_id / "summary.json", result)

    @staticmethod
    def _artifact_data_available(path: Path) -> bool:
        """Return whether an evidence file contains measurements, not a marker."""
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.name == "server.log":
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            return bool(text) and not text.lower().startswith(
                ("unavailable:", "server was not started")
            )
        if path.suffix == ".jsonl":
            measured_rows = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if not isinstance(row, dict):
                    return False
                if row.get("available") is not False:
                    measured_rows += 1
            return measured_rows > 0
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return False
            if isinstance(value, dict) and value.get("available") is False:
                return False
            if path.name == "server-command.json":
                if not isinstance(value, dict):
                    return False
                argv = value.get("argv")
                return isinstance(argv, list) and bool(argv)
            if path.name == "cleanup.json":
                if not isinstance(value, dict):
                    return False
                return all(value.get(field) is True for field in CLEANUP_REQUIRED_TRUE_FIELDS)
        return True

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSON object {path}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSON object {path}: expected an object")
        return value

    @staticmethod
    def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"Cannot read JSONL evidence {path}: {error}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL {path}:{line_number}: expected an object")
            rows.append(row)
        return rows

    def load_trial_result(self, trial_id: str) -> TrialResult | None:
        """Load a terminal trial for deterministic artifact-backed resume."""
        path = self.trials_dir / str(trial_id) / "summary.json"
        if not path.exists():
            return None
        try:
            result = TrialResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError(f"Invalid cached trial summary {path}: {error}") from error
        if not result.status.terminal:
            raise ValueError(f"Cached trial {trial_id} is not terminal: {result.status.value}")
        return result

    def ensure_trial_artifacts(self, result: TrialResult) -> dict[str, Any]:
        """Guarantee a complete trial layout without inventing measurements.

        The normal trial controller emits every raw file. If an alternate or
        interrupted controller omits one, this method writes an explicit
        ``available: false`` marker and records the degradation in
        ``artifact-status.json``. Consumers can therefore distinguish missing
        evidence from a measured zero.
        """
        directory = self.trial_dir(result.trial_id)
        base = Path("trials") / result.trial_id
        missing_before = {name for name in TRIAL_ARTIFACT_FILES if not (directory / name).exists()}
        reason = "artifact was not emitted by the trial controller"
        marker = {
            "record_type": "availability",
            "available": False,
            "reason": reason,
        }

        if "server-command.json" in missing_before:
            self.write_json(
                base / "server-command.json",
                {"available": False, "argv": None, "environment": None, "reason": reason},
            )
        if "params.json" in missing_before:
            self.write_json(base / "params.json", result.params)
        if "status.json" in missing_before:
            self.write_json(
                base / "status.json",
                {
                    "status": result.status.value,
                    "terminal": result.status.terminal,
                    "history": None,
                    "history_available": False,
                    "reason": reason,
                },
            )
        if "request-results.jsonl" in missing_before:
            self.write_jsonl(base / "request-results.jsonl", [marker])
        if "benchmark-raw.json" in missing_before:
            self.write_json(base / "benchmark-raw.json", marker)
        if "prometheus.jsonl" in missing_before:
            self.write_jsonl(base / "prometheus.jsonl", [marker])
        if "nvml.jsonl" in missing_before:
            self.write_jsonl(base / "nvml.jsonl", [marker])
        if "server.log" in missing_before:
            self.write_text(base / "server.log", f"unavailable: {reason}\n")
        if "cleanup.json" in missing_before:
            cleanup_status = getattr(result, "cleanup_status", None)
            if isinstance(cleanup_status, Mapping):
                self.write_json(base / "cleanup.json", dict(cleanup_status))
            else:
                self.write_json(base / "cleanup.json", marker)

        return self.seal_trial_artifacts(result, missing_before=missing_before)

    def seal_trial_artifacts(
        self,
        result: TrialResult,
        *,
        missing_before: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Rewrite derived status and checksums after the terminal summary is final."""
        directory = self.trial_dir(result.trial_id)
        base = Path("trials") / result.trial_id
        relative_paths = {name: str(base / name) for name in TRIAL_ARTIFACT_FILES}
        relative_paths[ARTIFACT_STATUS_FILE] = str(base / ARTIFACT_STATUS_FILE)
        relative_paths[ARTIFACT_INTEGRITY_FILE] = str(base / ARTIFACT_INTEGRITY_FILE)
        result.artifacts.update(relative_paths)
        # Rewrite only the terminal summary. status.json belongs to the lifecycle
        # state machine and must retain its transition history.
        self.save_trial_result(result)

        evidence_files = {
            "server-command.json",
            "request-results.jsonl",
            "benchmark-raw.json",
            "prometheus.jsonl",
            "nvml.jsonl",
            "server.log",
            "cleanup.json",
        }
        files: dict[str, dict[str, Any]] = {}
        for name in TRIAL_ARTIFACT_FILES:
            path = directory / name
            present = path.exists()
            size_bytes = path.stat().st_size if present else None
            data_available = present
            if name in evidence_files:
                data_available = self._artifact_data_available(path)
            files[name] = {
                "present": present,
                "data_available": data_available,
                "size_bytes": size_bytes,
            }
        unavailable_data = sorted(
            name for name, item in files.items() if not item["data_available"]
        )
        status = {
            "trial_id": result.trial_id,
            "complete_layout": all(item["present"] for item in files.values()),
            "degraded": bool(unavailable_data),
            "missing_before_finalize": sorted(missing_before),
            "unavailable_data": unavailable_data,
            "files": files,
        }
        self.write_json(base / ARTIFACT_STATUS_FILE, status)
        self._write_trial_integrity(result.trial_id)
        return status

    def record_artifact_finalizer_failure(self, result: TrialResult, reason: str) -> Path:
        """Append a post-controller FAILED transition without discarding history."""
        if result.status.value != "FAILED":
            raise ValueError("artifact finalizer status can only record a FAILED result")
        base = Path("trials") / result.trial_id
        path = self.trial_dir(result.trial_id) / "status.json"
        lifecycle = self._read_json_object(path)
        previous = lifecycle.get("status")
        history_value = lifecycle.get("history")
        history = list(history_value) if isinstance(history_value, list) else []
        history.append(
            {
                "previous": previous,
                "current": result.status.value,
                "monotonic_ns": time.perf_counter_ns(),
                "reason": reason,
                "source": "artifact_finalizer",
            }
        )
        lifecycle.update(
            {
                "status": result.status.value,
                "terminal": True,
                "history": history,
                "history_available": True,
            }
        )
        return self.write_json(base / "status.json", lifecycle)

    def _write_trial_integrity(self, trial_id: str) -> Path:
        """Seal every replay-relevant file with an exact size and SHA-256."""
        directory = self.trial_dir(trial_id)
        files: dict[str, dict[str, Any]] = {}
        actual_files = {
            path.relative_to(directory).as_posix(): path
            for path in directory.rglob("*")
            if path.is_file() and path.relative_to(directory).as_posix() != ARTIFACT_INTEGRITY_FILE
        }
        missing_core = sorted(set(SEALED_TRIAL_FILES) - set(actual_files))
        if missing_core:
            raise ValueError(
                f"Cannot seal missing trial artifacts {trial_id}: {', '.join(missing_core)}"
            )
        for name, path in sorted(actual_files.items()):
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        return self.write_json(
            Path("trials") / trial_id / ARTIFACT_INTEGRITY_FILE,
            {"schema_version": 1, "trial_id": trial_id, "files": files},
        )

    def validate_trial_integrity(self, trial_id: str) -> None:
        """Reject deleted, replaced, or byte-modified cached evidence."""
        directory = self.trial_dir(trial_id)
        integrity_path = directory / ARTIFACT_INTEGRITY_FILE
        if not integrity_path.is_file():
            raise ValueError(f"Trial {trial_id} has no {ARTIFACT_INTEGRITY_FILE}")
        integrity = self._read_json_object(integrity_path)
        if integrity.get("schema_version") != 1 or integrity.get("trial_id") != trial_id:
            raise ValueError(f"Trial {trial_id} has invalid artifact integrity metadata")
        files = integrity.get("files")
        if not isinstance(files, dict) or not set(SEALED_TRIAL_FILES).issubset(files):
            raise ValueError(f"Trial {trial_id} artifact integrity file set is invalid")
        actual_files = {
            path.relative_to(directory).as_posix(): path
            for path in directory.rglob("*")
            if path.is_file() and path.relative_to(directory).as_posix() != ARTIFACT_INTEGRITY_FILE
        }
        if set(files) != set(actual_files):
            added = sorted(set(actual_files) - set(files))
            missing = sorted(set(files) - set(actual_files))
            details = []
            if added:
                details.append("unsealed=" + ",".join(added))
            if missing:
                details.append("missing=" + ",".join(missing))
            raise ValueError(
                f"Trial {trial_id} artifact integrity file set mismatch: {'; '.join(details)}"
            )
        for name, path in sorted(actual_files.items()):
            expected = files.get(name)
            if not isinstance(expected, dict):
                raise ValueError(f"Trial {trial_id} artifact is missing: {name}")
            actual_size = path.stat().st_size
            actual_sha256 = sha256_file(path)
            if expected.get("size_bytes") != actual_size or expected.get("sha256") != actual_sha256:
                raise ValueError(f"Trial {trial_id} artifact checksum mismatch: {name}")

    @staticmethod
    def _require_equal(
        trial_id: str,
        description: str,
        actual: Any,
        expected: Any,
    ) -> None:
        if actual != expected:
            raise ValueError(
                f"Trial {trial_id} inconsistent {description}: "
                f"expected {expected!r}, found {actual!r}"
            )

    def validate_cached_trial(self, result: TrialResult, *, require_telemetry: bool) -> None:
        """Accept a known failed inconsistency only when it reproduces exactly."""
        failure = result.failure_reason if isinstance(result.failure_reason, Mapping) else {}
        expected = (
            failure.get("artifact_inconsistency")
            if failure.get("type") == "ARTIFACT_INCONSISTENT"
            else None
        )
        try:
            self._validate_cached_trial_strict(result, require_telemetry=require_telemetry)
        except ValueError as error:
            if isinstance(expected, str) and str(error) == expected:
                return
            raise
        if isinstance(expected, str):
            raise ValueError(
                f"Trial {result.trial_id} no longer reproduces its recorded artifact inconsistency"
            )

    def _validate_cached_trial_strict(
        self, result: TrialResult, *, require_telemetry: bool
    ) -> None:
        """Cross-check checksummed raw evidence against its terminal summary."""
        trial_id = result.trial_id
        directory = self.trial_dir(trial_id)
        self.validate_trial_integrity(trial_id)

        expected_provenance = trial_provenance(trial_id, result.method)
        manifest_path = self.root / "manifest.json"
        requires_provenance = any(
            value is not None
            for value in (result.phase, result.source_method, result.source_trial_id)
        )
        if manifest_path.is_file():
            manifest = self._read_json_object(manifest_path)
            try:
                requires_provenance = (
                    requires_provenance or int(manifest.get("artifact_schema_version", 0)) >= 5
                )
            except (TypeError, ValueError):
                raise ValueError("Invalid manifest artifact_schema_version") from None
        if requires_provenance:
            for field, expected_value in expected_provenance.items():
                self._require_equal(
                    trial_id,
                    f"provenance {field}",
                    getattr(result, field),
                    expected_value,
                )

        params = self._read_json_object(directory / "params.json")
        self._require_equal(trial_id, "params.json", params, result.params)

        lifecycle = self._read_json_object(directory / "status.json")
        self._require_equal(
            trial_id,
            "status.json status",
            lifecycle.get("status"),
            result.status.value,
        )
        history = lifecycle.get("history")
        if isinstance(history, list) and history:
            last = history[-1]
            if not isinstance(last, dict):
                raise ValueError(f"Trial {trial_id} has invalid lifecycle history")
            self._require_equal(
                trial_id,
                "status.json terminal transition",
                last.get("current"),
                result.status.value,
            )

        cleanup = self._read_json_object(directory / "cleanup.json")
        cleanup_status = getattr(result, "cleanup_status", None)
        if isinstance(cleanup_status, Mapping):
            self._require_equal(trial_id, "cleanup.json", cleanup, dict(cleanup_status))
        if result.selectable and not all(
            cleanup.get(field) is True for field in CLEANUP_REQUIRED_TRUE_FIELDS
        ):
            raise ValueError(f"Trial {trial_id} has unverified process/GPU cleanup")

        requests = self._read_jsonl_objects(directory / "request-results.jsonl")
        raw = self._read_json_object(directory / "benchmark-raw.json")
        raw_available = raw.get("available") is not False
        if raw_available:
            raw_requests = raw.get("request_results")
            if not isinstance(raw_requests, list) or not all(
                isinstance(row, dict) for row in raw_requests
            ):
                raise ValueError(f"Trial {trial_id} benchmark raw requests are invalid")
            self._require_equal(
                trial_id,
                "request-results.jsonl and benchmark-raw.json",
                requests,
                raw_requests,
            )
            identifiers = [row.get("request_id") for row in requests]
            if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
                raise ValueError(f"Trial {trial_id} has a request without a request_id")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"Trial {trial_id} has duplicate request_id evidence")

            aggregate = raw.get("aggregate")
            if not isinstance(aggregate, dict):
                raise ValueError(f"Trial {trial_id} benchmark aggregate is invalid")
            for key, value in aggregate.items():
                self._require_equal(
                    trial_id,
                    f"client.{key}",
                    result.client.get(key),
                    value,
                )
            successful = [row for row in requests if row.get("status") == "success"]
            derived = {
                "num_requests": len(requests),
                "completed": len(successful),
                "failed": len(requests) - len(successful),
                "total_input_tokens": sum(row.get("input_tokens", 0) for row in successful),
                "total_output_tokens": sum(row.get("output_tokens", 0) for row in successful),
            }
            for key, value in derived.items():
                self._require_equal(
                    trial_id, f"derived client.{key}", result.client.get(key), value
                )
        elif result.selectable:
            raise ValueError(
                f"Trial {trial_id} selectable summary has unavailable benchmark raw data"
            )

        prometheus = self._read_jsonl_objects(directory / "prometheus.jsonl")
        nvml = self._read_jsonl_objects(directory / "nvml.jsonl")
        if result.engine.get("sample_count") is not None:
            self._require_equal(
                trial_id,
                "engine sample_count",
                result.engine.get("sample_count"),
                len(prometheus),
            )
        if result.gpu.get("sample_count") is not None:
            self._require_equal(
                trial_id,
                "GPU sample_count",
                result.gpu.get("sample_count"),
                len(nvml),
            )

    def validate_trial_artifacts(
        self,
        trial_id: str,
        require_telemetry: bool = True,
        *,
        require_available: bool = False,
        required_evidence: Iterable[str] | None = None,
    ) -> None:
        """Fail before optimizer submission when required raw evidence is absent."""
        directory = self.trial_dir(trial_id)
        required = {
            "server-command.json",
            "params.json",
            "status.json",
            "request-results.jsonl",
            "benchmark-raw.json",
            "server.log",
            "cleanup.json",
            "summary.json",
        }
        if require_telemetry:
            required.update({"prometheus.jsonl", "nvml.jsonl"})
        missing = sorted(name for name in required if not (directory / name).exists())
        if missing:
            raise ValueError(f"Trial {trial_id} has incomplete artifacts: {', '.join(missing)}")
        if require_available:
            evidence = set(
                required_evidence
                or {
                    "server-command.json",
                    "request-results.jsonl",
                    "benchmark-raw.json",
                    "server.log",
                    "cleanup.json",
                }
            )
            evidence.add("cleanup.json")
            if require_telemetry:
                evidence.update({"prometheus.jsonl", "nvml.jsonl"})
            unavailable = sorted(
                name for name in evidence if not self._artifact_data_available(directory / name)
            )
            if unavailable:
                raise ValueError(
                    f"Trial {trial_id} has unavailable evidence: {', '.join(unavailable)}"
                )

    def _experiment_trial_ids(self) -> list[str]:
        """Return flat trial directories, rejecting unanchored entries."""
        if not self.trials_dir.is_dir():
            raise ValueError(f"Experiment {self.root.name} has no trials directory")
        trial_ids: list[str] = []
        for entry in sorted(self.trials_dir.iterdir(), key=lambda item: item.name):
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError(
                    f"Experiment {self.root.name} has unexpected trials entry: {entry.name}"
                )
            trial_ids.append(entry.name)
        return trial_ids

    def _experiment_sealed_files(self) -> dict[str, Path]:
        """Enumerate non-trial files plus each child integrity anchor."""
        files: dict[str, Path] = {}
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    f"Experiment {self.root.name} contains unsupported symlink: "
                    f"{path.relative_to(self.root).as_posix()}"
                )
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            name = relative.as_posix()
            if name == EXPERIMENT_INTEGRITY_FILE:
                continue
            if relative.parts and relative.parts[0] == "trials":
                if len(relative.parts) == 3 and relative.name == ARTIFACT_INTEGRITY_FILE:
                    files[name] = path
                continue
            files[name] = path
        return files

    @staticmethod
    def _compact_simulation_result(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {key: value.get(key) for key in ("policy_name", "seed", "metrics") if key in value}

    def compact_scheduler_ablation(self, scheduler: Mapping[str, Any]) -> dict[str, Any]:
        """Keep aggregate scheduler evidence in summary and reference raw rows by hash."""
        raw_path = self.aggregate_dir / "scheduler-ablation.json"
        if not raw_path.is_file():
            raise ValueError("Cannot compact scheduler ablation without its raw artifact")
        compact: dict[str, Any] = {
            "schema_version": 1,
            "raw_artifact": raw_path.relative_to(self.root).as_posix(),
            "raw_size_bytes": raw_path.stat().st_size,
            "raw_sha256": sha256_file(raw_path),
            "has_negative_result": bool(scheduler.get("has_negative_result", False)),
            "negative_gain_conditions": list(
                scheduler.get("negative_gain_conditions", [])
                if isinstance(scheduler.get("negative_gain_conditions"), list)
                else []
            ),
        }
        for label in ("calibration", "held_out"):
            raw_section = scheduler.get(label)
            if not isinstance(raw_section, Mapping):
                continue
            baselines = raw_section.get("fixed_baselines")
            compact[label] = {
                "trace_name": raw_section.get("trace_name"),
                "best_fixed_budget": raw_section.get("best_fixed_budget"),
                "goodput_gain_vs_best": raw_section.get("goodput_gain_vs_best"),
                "negative_gain_conditions": list(
                    raw_section.get("negative_gain_conditions", [])
                    if isinstance(raw_section.get("negative_gain_conditions"), list)
                    else []
                ),
                "adaptive": self._compact_simulation_result(raw_section.get("adaptive")),
                "fixed_baselines": {
                    str(budget): self._compact_simulation_result(result)
                    for budget, result in sorted(
                        (baselines.items() if isinstance(baselines, Mapping) else []),
                        key=lambda item: str(item[0]),
                    )
                },
            }
        return compact

    @staticmethod
    def _negative_results_markdown(conditions: list[dict[str, Any]]) -> str:
        lines = [
            "# Scheduler negative/no-benefit conditions",
            "",
            "| Trace | Metric | Adaptive | Best fixed | Budget | Relative gain | Explanation |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for condition in conditions:
            lines.append(
                "| {trace} | {metric} | {adaptive} | {fixed} | {budget} | {gain} | "
                "{explanation} |".format(
                    trace=condition.get("trace_name", "unknown"),
                    metric=condition.get("metric", "unknown"),
                    adaptive=condition.get("adaptive_value", "unavailable"),
                    fixed=condition.get("fixed_value", "unavailable"),
                    budget=condition.get("fixed_budget", "unavailable"),
                    gain=condition.get("relative_gain", "unavailable"),
                    explanation=str(condition.get("explanation", "unavailable")).replace(
                        "|", "\\|"
                    ),
                )
            )
        if not conditions:
            lines.append("| none | none | unavailable | unavailable | unavailable | 0 | None |")
        return "\n".join(lines) + "\n"

    def _preflight_experiment_inputs(self) -> None:
        """Reject an incomplete root before writing any derived attestation bytes."""
        missing: list[str] = []
        invalid: list[str] = []
        for name in REQUIRED_EXPERIMENT_INPUT_FILES:
            path = self.root / name
            if not path.exists():
                missing.append(name)
                continue
            if path.is_symlink() or not path.is_file():
                invalid.append(name)
                continue
            relative_parent = path.parent
            while relative_parent != self.root:
                if relative_parent.is_symlink():
                    invalid.append(name)
                    break
                relative_parent = relative_parent.parent
        for name in GENERATED_EXPERIMENT_FILES:
            path = self.root / name
            if path.exists() and (path.is_symlink() or not path.is_file()):
                invalid.append(name)
            relative_parent = path.parent
            while relative_parent != self.root:
                if relative_parent.exists() and relative_parent.is_symlink():
                    invalid.append(name)
                    break
                relative_parent = relative_parent.parent
        if missing or invalid:
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if invalid:
                details.append("not-regular=" + ",".join(sorted(set(invalid))))
            raise ValueError(
                f"Experiment {self.root.name} required input preflight failed: "
                + "; ".join(details)
            )
        # Reject unsupported links anywhere in the existing root before any
        # derived view is written. Regular extras are included in the seal.
        self._experiment_sealed_files()

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        converted = float(value)
        return converted if math.isfinite(converted) else None

    def _capacity_rate_semantics(
        self,
        summary: Mapping[str, Any],
        capacity_results: list[TrialResult],
    ) -> dict[str, Any]:
        """Derive empirical capacity rates from sealed child summaries."""
        capacity_value = summary.get("capacity_sweep")
        capacity = capacity_value if isinstance(capacity_value, Mapping) else {}
        points_value = capacity.get("points")
        points = (
            [dict(item) for item in points_value if isinstance(item, Mapping)]
            if isinstance(points_value, list)
            else []
        )
        if points_value is not None and not isinstance(points_value, list):
            raise ValueError("Capacity summary points must be a list")
        if isinstance(points_value, list) and len(points) != len(points_value):
            raise ValueError("Capacity summary contains a non-object point")
        points_by_trial = {
            str(point["trial_id"]): point
            for point in points
            if isinstance(point.get("trial_id"), str)
        }
        if len(points_by_trial) != len(points):
            raise ValueError("Capacity summary contains a missing or duplicate trial_id")
        capacity_trial_ids = {result.trial_id for result in capacity_results}
        if set(points_by_trial) != capacity_trial_ids:
            raise ValueError(
                "Capacity summary/trial set mismatch: "
                f"summary={sorted(points_by_trial)}, trials={sorted(capacity_trial_ids)}"
            )
        has_explicit_rates = bool(points) and all(
            "target_offered_requests_per_sec" in point
            and "empirical_scheduled_requests_per_sec" in point
            for point in points
        )
        rows: list[dict[str, Any]] = []
        grouped: dict[float, list[float]] = {}
        for result in capacity_results:
            match = re.fullmatch(r"capacity-rate-(.+)-repeat-(\d+)", result.trial_id)
            parsed_target: float | None = None
            if match is not None:
                try:
                    parsed_target = float(match.group(1))
                except ValueError:
                    parsed_target = None
            client_target = self._finite_number(result.client.get("offered_requests_per_sec"))
            target = parsed_target if parsed_target is not None else client_target
            empirical = self._finite_number(
                result.client.get("empirical_scheduled_requests_per_sec")
            )
            point = points_by_trial.get(result.trial_id)
            if point is not None and target is not None:
                for key in (
                    "target_offered_requests_per_sec",
                    "offered_requests_per_sec",
                    "measured_offered_requests_per_sec",
                ):
                    recorded = self._finite_number(point.get(key))
                    if key in point and recorded is None:
                        raise ValueError(f"Capacity point {result.trial_id} has non-finite {key}")
                    if recorded is not None and recorded != target:
                        raise ValueError(
                            f"Capacity point {result.trial_id} has inconsistent {key}: "
                            f"expected {target!r}, found {recorded!r}"
                        )
                recorded_empirical = self._finite_number(
                    point.get("empirical_scheduled_requests_per_sec")
                )
                if (
                    "empirical_scheduled_requests_per_sec" in point
                    and recorded_empirical != empirical
                ):
                    raise ValueError(
                        f"Capacity point {result.trial_id} has inconsistent empirical "
                        f"scheduled rate: expected {empirical!r}, found {recorded_empirical!r}"
                    )
            if target is not None and empirical is not None:
                grouped.setdefault(target, []).append(empirical)
            rows.append(
                {
                    "trial_id": result.trial_id,
                    "status": result.status.value,
                    "target_offered_requests_per_sec": target,
                    "empirical_scheduled_requests_per_sec": empirical,
                    "achieved_requests_per_sec": self._finite_number(
                        result.client.get("achieved_requests_per_sec")
                    ),
                }
            )
        by_target = [
            {
                "target_offered_requests_per_sec": target,
                "measured_count": len(values),
                "median_empirical_scheduled_requests_per_sec": statistics.median(values),
                "min_empirical_scheduled_requests_per_sec": min(values),
                "max_empirical_scheduled_requests_per_sec": max(values),
            }
            for target, values in sorted(grouped.items())
        ]
        return {
            "schema_version": 1,
            "summary_capacity_schema": (
                "target_empirical_v2" if has_explicit_rates else "legacy_target_alias_v1"
            ),
            "legacy_aggregate_column_semantics": {
                "offered_requests_per_sec": "target offered rate",
                "measured_offered_requests_per_sec": (
                    "legacy target-rate alias; not an empirical measurement"
                ),
            },
            "empirical_source": (
                "sealed trial summary client.empirical_scheduled_requests_per_sec"
            ),
            "trial_count": len(rows),
            "trials": rows,
            "by_target_rate": by_target,
        }

    def _write_experiment_views(self) -> dict[str, Any]:
        """Create compact, lineage, negative-result, and semantic-audit views."""
        self._preflight_experiment_inputs()
        summary_path = self.root / "summary.json"
        scheduler_path = self.aggregate_dir / "scheduler-ablation.json"
        summary = self._read_json_object(summary_path)
        scheduler = self._read_json_object(scheduler_path)
        compact = self.compact_scheduler_ablation(scheduler)
        conditions_value = scheduler.get("negative_gain_conditions")
        conditions = (
            [dict(item) for item in conditions_value if isinstance(item, Mapping)]
            if isinstance(conditions_value, list)
            else []
        )
        negative_view = {
            "schema_version": 1,
            "raw_artifact": compact["raw_artifact"],
            "raw_size_bytes": compact["raw_size_bytes"],
            "raw_sha256": compact["raw_sha256"],
            "has_negative_result": bool(conditions),
            "condition_count": len(conditions),
            "conditions": conditions,
        }

        manifest = self._read_json_object(self.root / "manifest.json")
        root_semantic_checks: dict[str, bool] = {
            "manifest_experiment_id": manifest.get("experiment_id") == self.root.name,
            "summary_experiment_id": summary.get("experiment_id") == self.root.name,
            "trace_sha256": sha256_file(self.root / "trace.jsonl") == manifest.get("trace_sha256"),
            "holdout_trace_sha256": sha256_file(self.root / "holdout-trace.jsonl")
            == manifest.get("holdout_trace_sha256"),
            "trace_sidecar": str(manifest.get("trace_sha256"))
            in (self.root / "trace.sha256").read_text(encoding="utf-8"),
            "holdout_trace_sidecar": str(manifest.get("holdout_trace_sha256"))
            in (self.root / "holdout-trace.sha256").read_text(encoding="utf-8"),
        }
        summary_manifest = summary.get("manifest")
        identity_fields = (
            "experiment_id",
            "model",
            "model_revision",
            "tokenizer",
            "trace_sha256",
            "holdout_trace_sha256",
            "search_space_sha256",
            "experiment_config_sha256",
            "seed",
            "source_commit",
            "source_tree_sha256",
            "dirty_worktree",
        )
        root_semantic_checks["summary_manifest_identity"] = isinstance(
            summary_manifest, Mapping
        ) and all(summary_manifest.get(field) == manifest.get(field) for field in identity_fields)
        root_semantic_checks["summary_manifest_exact"] = summary_manifest == manifest
        root_semantic_checks["summary_scheduler_raw"] = (
            summary.get("scheduler_ablation") == scheduler
        )
        root_semantic_checks["compact_summary_identity"] = (
            summary.get("experiment_id") == self.root.name and summary_manifest == manifest
        )
        root_semantic_checks["compact_scheduler_reference"] = (
            compact.get("raw_artifact") == "aggregate/scheduler-ablation.json"
            and compact.get("raw_size_bytes") == scheduler_path.stat().st_size
            and compact.get("raw_sha256") == sha256_file(scheduler_path)
        )
        failed_root_checks = sorted(
            name for name, passed in root_semantic_checks.items() if not passed
        )
        if failed_root_checks:
            raise ValueError(
                "Experiment root evidence is inconsistent: " + ", ".join(failed_root_checks)
            )
        require_telemetry = bool(
            manifest.get("telemetry", {}).get("enabled", False)
            if isinstance(manifest.get("telemetry"), Mapping)
            else False
        )
        status_counts: dict[str, int] = {}
        selectable_count = 0
        legacy_provenance: list[str] = []
        anchors: list[dict[str, Any]] = []
        trial_lineage: list[dict[str, Any]] = []
        capacity_results: list[TrialResult] = []
        recorded_provenance_count = 0
        derived_repeat_holdout_count = 0
        trial_ids = self._experiment_trial_ids()
        for trial_id in trial_ids:
            result = self.load_trial_result(trial_id)
            if result is None:
                raise ValueError(f"Trial {trial_id} has no terminal summary")
            self.validate_cached_trial(result, require_telemetry=require_telemetry)
            status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
            selectable_count += int(result.selectable)
            canonical = trial_provenance(trial_id, result.method)
            provenance_recorded = result.phase is not None and result.source_method is not None
            if not provenance_recorded:
                legacy_provenance.append(trial_id)
            else:
                recorded_provenance_count += 1
            phase = result.phase if provenance_recorded else canonical["phase"]
            source_method = (
                result.source_method if provenance_recorded else canonical["source_method"]
            )
            source_trial_id = (
                result.source_trial_id if provenance_recorded else canonical["source_trial_id"]
            )
            if not provenance_recorded and phase in {"repeat", "holdout"}:
                derived_repeat_holdout_count += 1
            if canonical["phase"] == "capacity":
                capacity_results.append(result)
            anchor = self.trials_dir / trial_id / ARTIFACT_INTEGRITY_FILE
            anchor_record = {
                "trial_id": trial_id,
                "path": anchor.relative_to(self.root).as_posix(),
                "size_bytes": anchor.stat().st_size,
                "sha256": sha256_file(anchor),
            }
            anchors.append(anchor_record)
            trial_lineage.append(
                {
                    **anchor_record,
                    "recorded_method": result.method,
                    "phase": phase,
                    "source_method": source_method,
                    "source_trial_id": source_trial_id,
                    "provenance_kind": (
                        "recorded" if provenance_recorded else "derived_from_trial_id"
                    ),
                }
            )

        capacity_semantics = self._capacity_rate_semantics(summary, capacity_results)

        # No experiment-level bytes are changed until every child checksum and
        # semantic cross-check has passed. A failed first attestation therefore
        # cannot leave a partially rewritten legacy/formal directory behind.
        self.write_json("aggregate/scheduler-negative-results.json", negative_view)
        self.write_text(
            "report/scheduler-negative-results.md",
            self._negative_results_markdown(conditions),
        )

        root_inputs: dict[str, dict[str, Any]] = {}
        for name in (
            "manifest.json",
            "experiment.yaml",
            "trace.jsonl",
            "trace.sha256",
            "holdout-trace.jsonl",
            "holdout-trace.sha256",
            "summary.json",
            "aggregate/scheduler-ablation.json",
            "report/report.md",
            "report/report.html",
            "report/plot-manifest.json",
        ):
            path = self.root / name
            root_inputs[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        lineage = {
            "schema_version": 1,
            "experiment_id": self.root.name,
            "root_inputs": root_inputs,
            "trial_integrity_anchors": anchors,
            "trials": trial_lineage,
        }
        audit = {
            "schema_version": 1,
            "experiment_id": self.root.name,
            "trial_count": len(trial_ids),
            "trial_integrity_validated": len(trial_ids),
            "trial_semantic_validated": len(trial_ids),
            "require_telemetry": require_telemetry,
            "selectable_trial_count": selectable_count,
            "status_counts": status_counts,
            "legacy_provenance_trial_count": len(legacy_provenance),
            "legacy_provenance_trial_ids": legacy_provenance,
            "recorded_provenance_trial_count": recorded_provenance_count,
            "derived_provenance_trial_count": len(legacy_provenance),
            "derived_repeat_holdout_trial_count": derived_repeat_holdout_count,
            "scheduler_negative_condition_count": len(conditions),
            "capacity_rate_semantics": capacity_semantics,
            "root_semantic_checks": root_semantic_checks,
        }
        self.write_json("lineage.json", lineage)
        self.write_json("experiment-audit.json", audit)
        compact_summary = dict(summary)
        compact_summary["scheduler_ablation"] = compact
        compact_summary["capacity_rate_semantics"] = capacity_semantics
        compact_summary["experiment_attestation"] = {
            "schema_version": 1,
            "original_summary": {
                "path": "summary.json",
                "size_bytes": summary_path.stat().st_size,
                "sha256": sha256_file(summary_path),
            },
            "lineage": "lineage.json",
            "audit": "experiment-audit.json",
            "scheduler_negative_results": "aggregate/scheduler-negative-results.json",
            "scheduler_negative_report": "report/scheduler-negative-results.md",
        }
        self.write_json(SUMMARY_COMPACT_FILE, compact_summary)
        return audit

    def seal_experiment_artifacts(
        self,
        *,
        attestation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Seal every non-trial artifact and each validated trial-integrity anchor."""
        trial_ids = self._experiment_trial_ids()
        for trial_id in trial_ids:
            self.validate_trial_integrity(trial_id)
        files = self._experiment_sealed_files()
        missing = sorted(set(REQUIRED_EXPERIMENT_FILES) - set(files))
        if missing:
            raise ValueError(
                f"Cannot seal missing experiment artifacts {self.root.name}: " + ", ".join(missing)
            )

        integrity_path = self.root / EXPERIMENT_INTEGRITY_FILE
        attestations: list[dict[str, Any]] = []
        previous_sha256: str | None = None
        if integrity_path.is_file():
            previous_sha256 = sha256_file(integrity_path)
            previous = self._read_json_object(integrity_path)
            value = previous.get("attestations")
            if isinstance(value, list):
                attestations = [dict(item) for item in value if isinstance(item, Mapping)]
        if attestation is not None:
            supplied = dict(attestation)
            attestation_kind = supplied.pop("attestation_kind", supplied.pop("kind", "unspecified"))
            repository = Path(__file__).resolve().parents[3]
            tool_commit, tool_dirty, _ = git_state(repository)
            tool_tree = source_tree_sha256(repository)
            manifest = self._read_json_object(self.root / "manifest.json")
            record = {
                **supplied,
                "schema_version": 1,
                "attestation_kind": attestation_kind,
                "attested_at_utc": supplied.pop("attested_at_utc", utc_now_iso()),
                "measurement_source_commit": manifest.get("source_commit"),
                "measurement_source_tree_sha256": manifest.get("source_tree_sha256"),
                "measurement_dirty_worktree": manifest.get("dirty_worktree"),
                "attestation_source_commit": supplied.pop("attestation_source_commit", tool_commit),
                "attestation_source_tree_sha256": supplied.pop(
                    "attestation_source_tree_sha256", tool_tree
                ),
                "attestation_dirty_worktree": supplied.pop(
                    "attestation_dirty_worktree", tool_dirty
                ),
                "experiment_config_sha256": manifest.get("experiment_config_sha256"),
                "trace_sha256": manifest.get("trace_sha256"),
                "holdout_trace_sha256": manifest.get("holdout_trace_sha256"),
            }
            if previous_sha256 is not None:
                record["previous_integrity_sha256"] = previous_sha256
            attestations.append(record)

        payload = {
            "schema_version": 1,
            "experiment_id": self.root.name,
            "files": {
                name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(files.items())
            },
            "trial_integrity_anchors": [
                f"trials/{trial_id}/{ARTIFACT_INTEGRITY_FILE}" for trial_id in trial_ids
            ],
            "attestations": attestations,
        }
        self.write_json(EXPERIMENT_INTEGRITY_FILE, payload)
        return payload

    def validate_experiment_integrity(self) -> dict[str, Any]:
        """Reject added, deleted, or byte-modified root and child artifacts."""
        integrity_path = self.root / EXPERIMENT_INTEGRITY_FILE
        if not integrity_path.is_file():
            raise ValueError(f"Experiment {self.root.name} has no {EXPERIMENT_INTEGRITY_FILE}")
        integrity = self._read_json_object(integrity_path)
        if integrity.get("schema_version") != 1 or integrity.get("experiment_id") != self.root.name:
            raise ValueError(f"Experiment {self.root.name} has invalid integrity metadata")
        files = integrity.get("files")
        if not isinstance(files, dict) or not set(REQUIRED_EXPERIMENT_FILES).issubset(files):
            raise ValueError(f"Experiment {self.root.name} integrity file set is invalid")
        actual_files = self._experiment_sealed_files()
        if set(files) != set(actual_files):
            added = sorted(set(actual_files) - set(files))
            missing = sorted(set(files) - set(actual_files))
            details = []
            if added:
                details.append("unsealed=" + ",".join(added))
            if missing:
                details.append("missing=" + ",".join(missing))
            raise ValueError(
                f"Experiment {self.root.name} integrity file set mismatch: " + "; ".join(details)
            )

        trial_ids = self._experiment_trial_ids()
        anchors = [f"trials/{trial_id}/{ARTIFACT_INTEGRITY_FILE}" for trial_id in trial_ids]
        if integrity.get("trial_integrity_anchors") != anchors:
            raise ValueError(f"Experiment {self.root.name} trial anchor set is invalid")
        for trial_id in trial_ids:
            self.validate_trial_integrity(trial_id)
        for name, path in sorted(actual_files.items()):
            expected = files.get(name)
            if not isinstance(expected, Mapping):
                raise ValueError(f"Experiment {self.root.name} artifact is missing: {name}")
            if expected.get("size_bytes") != path.stat().st_size or expected.get(
                "sha256"
            ) != sha256_file(path):
                raise ValueError(f"Experiment {self.root.name} artifact checksum mismatch: {name}")
        return integrity

    def attest_experiment_artifacts(
        self,
        *,
        attestation: Mapping[str, Any],
        reseal: bool = False,
        validate_existing: bool = True,
    ) -> dict[str, Any]:
        """Build derived audit views, then seal and verify the complete experiment."""
        integrity_path = self.root / EXPERIMENT_INTEGRITY_FILE
        if integrity_path.is_file() and not reseal:
            self.validate_experiment_integrity()
            return {"already_sealed": True, "audit": None, "integrity": integrity_path}
        if integrity_path.is_file() and validate_existing:
            self.validate_experiment_integrity()
        audit = self._write_experiment_views()
        integrity = self.seal_experiment_artifacts(attestation=attestation)
        self.validate_experiment_integrity()
        return {"already_sealed": False, "audit": audit, "integrity": integrity}
