"""Atomic artifact layout and integrity checks for reproducible experiments."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel

from .manifest import sha256_file
from .models import ExperimentSpec, TrialResult

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
SEALED_TRIAL_FILES = (*TRIAL_ARTIFACT_FILES, ARTIFACT_STATUS_FILE)
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
