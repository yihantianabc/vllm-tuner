"""Sealed, zero-GPU v2 boundary analysis for completed M1 capacity sweeps."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from vllm_tuner.experiment.artifacts import ArtifactStore
from vllm_tuner.experiment.manifest import (
    git_state,
    sha256_file,
    sha256_json,
    source_tree_sha256,
)
from vllm_tuner.experiment.models import utc_now_iso

from .m1_capacity_analysis import CapacitySweepAnalysis
from .m1_capacity_boundaries import (
    CAPACITY_BOUNDARY_VERSION,
    CapacityBoundaryAnalysis,
    derive_capacity_boundaries,
)
from .m1_capacity_integrity import (
    M1_CAPACITY_INTEGRITY_FILE,
    seal_m1_capacity_artifacts,
    validate_m1_capacity_artifacts,
)

M1_CAPACITY_REANALYSIS_SCHEMA = CAPACITY_BOUNDARY_VERSION
MANIFEST_FILE = "manifest.json"
EXPERIMENT_FILE = "experiment.json"
SUMMARY_FILE = "summary.json"
STATUS_FILE = "status.json"
REPORT_FILE = "report/m1-capacity-boundaries.md"
RUNNER_LOG_FILE = "runner.log"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REQUIRED_CONTEXT_TOKENS = {8_192, 16_384, 32_768}
_EXPECTED_SOURCE_KNEE_FAILURES = {
    "capacity_knees_accepted",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object with string keys")
    return value


class M1CapacityBoundaryRunner:
    """Derive and seal v2 boundaries without mutating or rerunning v1 evidence."""

    def __init__(
        self,
        artifact_root: str | Path,
        source_experiment_id: str,
        experiment_id: str,
        *,
        repository: str | Path,
        resume: bool = False,
        require_clean_source: bool = True,
        analysis_source_identity: Optional[tuple[str, str]] = None,
    ) -> None:
        for name, value in (
            ("source_experiment_id", source_experiment_id),
            ("experiment_id", experiment_id),
        ):
            if _PORTABLE_ID.fullmatch(value) is None:
                raise ValueError(f"{name} must be one portable path component")
        if source_experiment_id == experiment_id:
            raise ValueError("boundary artifact must not overwrite its source experiment")
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.source_experiment_id = source_experiment_id
        self.experiment_id = experiment_id
        self.repository = Path(repository).expanduser().resolve()
        self.resume = resume
        self.require_clean_source = require_clean_source
        self.analysis_source_identity = analysis_source_identity
        self.source_root = self.artifact_root / source_experiment_id
        self.store = ArtifactStore(self.artifact_root, experiment_id)

    def _analysis_identity(self) -> tuple[str, str]:
        if self.analysis_source_identity is not None:
            return self.analysis_source_identity
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if commit is None or tree is None:
            raise ValueError("M1 boundary analysis requires a Git source identity")
        if self.require_clean_source and dirty:
            raise ValueError("M1 boundary analysis requires one clean committed source identity")
        return commit, tree

    def _load_source(self) -> dict[str, Any]:
        seal = validate_m1_capacity_artifacts(self.source_root)
        attestation = _mapping(seal.get("attestation"), "source seal attestation")
        manifest = _read_json(self.source_root / MANIFEST_FILE)
        experiment = _read_json(self.source_root / EXPERIMENT_FILE)
        summary = _read_json(self.source_root / SUMMARY_FILE)
        for description, value in (
            ("source seal", seal.get("experiment_id")),
            ("source manifest", manifest.get("experiment_id")),
            ("source summary", summary.get("experiment_id")),
        ):
            if value != self.source_experiment_id:
                raise ValueError(f"{description} experiment identity mismatch")
        if (
            summary.get("schema_version") != "longctx-m1-capacity.v1"
            or summary.get("project_line") != "longctx-v5"
            or summary.get("milestone") != "M1"
            or summary.get("experiment_kind") != "capacity-sweep"
            or summary.get("evidence_role") != "formal"
        ):
            raise ValueError("source is not one formal long-context v5 M1 capacity sweep")
        if (
            attestation.get("project_line") != "longctx-v5"
            or attestation.get("milestone") != "M1"
            or attestation.get("experiment_kind") != "capacity-sweep"
            or attestation.get("evidence_role") != "formal"
        ):
            raise ValueError("source seal does not attest formal long-context v5 M1 evidence")

        analysis_payload = _mapping(summary.get("analysis"), "source summary.analysis")
        analysis = CapacitySweepAnalysis.model_validate_json(
            json.dumps(analysis_payload, sort_keys=True, allow_nan=False)
        )
        execution = _mapping(summary.get("execution"), "source summary.execution")
        acceptance = _mapping(summary.get("acceptance"), "source summary.acceptance")
        checks = _mapping(acceptance.get("checks"), "source summary.acceptance.checks")
        knee_policy = _mapping(experiment.get("knee_policy"), "source experiment.knee_policy")
        if knee_policy.get("below_lowest_result") != "left-censored-below-lowest-load":
            raise ValueError("source did not preregister the required left-censored result")
        if knee_policy.get("no_overload_result") != "right-censored-above-highest-load":
            raise ValueError("source did not preregister the required right-censored result")
        if attestation.get("capacity_accepted") is not acceptance.get("passed"):
            raise ValueError("source seal and summary disagree about v1 acceptance")
        return {
            "seal": seal,
            "attestation": dict(attestation),
            "manifest": manifest,
            "experiment": experiment,
            "summary": summary,
            "analysis": analysis,
            "execution": dict(execution),
            "acceptance": dict(acceptance),
            "checks": dict(checks),
            "knee_policy": dict(knee_policy),
            "seal_sha256": sha256_file(self.source_root / M1_CAPACITY_INTEGRITY_FILE),
            "summary_sha256": sha256_file(self.source_root / SUMMARY_FILE),
        }

    def _manifest(
        self,
        source: Mapping[str, Any],
        boundaries: CapacityBoundaryAnalysis,
        analysis_commit: str,
        analysis_tree: str,
    ) -> dict[str, Any]:
        source_manifest = _mapping(source["manifest"], "source manifest")
        threshold_policy = boundaries.threshold_policy.model_dump(mode="json")
        boundary_policy = {
            "schema_version": CAPACITY_BOUNDARY_VERSION,
            "numeric_thresholds_modified": False,
            "threshold_policy": threshold_policy,
            "threshold_policy_sha256": sha256_json(threshold_policy),
            "service_boundary_rule": boundaries.service_boundary_rule,
            "saturation_boundary_rule": boundaries.saturation_boundary_rule,
            "below_lowest_result": boundaries.below_lowest_result,
            "no_breach_or_overload_result": boundaries.no_breach_or_overload_result,
        }
        return {
            "schema_version": M1_CAPACITY_REANALYSIS_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M1",
            "experiment_kind": "capacity-boundary-reanalysis",
            "evidence_role": "formal-derived",
            "experiment_id": self.experiment_id,
            "created_at": utc_now_iso(),
            "analysis_source_commit": analysis_commit,
            "analysis_source_tree_sha256": analysis_tree,
            "gpu_runs_executed": 0,
            "source_artifact": {
                "experiment_id": self.source_experiment_id,
                "root": str(self.source_root),
                "schema_version": "longctx-m1-capacity.v1",
                "source_commit": source_manifest.get("source_commit"),
                "source_tree_sha256": source_manifest.get("source_tree_sha256"),
                "integrity_sha256": source["seal_sha256"],
                "summary_sha256": source["summary_sha256"],
                "capacity_accepted_v1": source["acceptance"].get("passed"),
            },
            "boundary_policy": boundary_policy,
        }

    @staticmethod
    def _manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: value.get(name)
            for name in (
                "schema_version",
                "project_line",
                "milestone",
                "experiment_kind",
                "evidence_role",
                "experiment_id",
                "analysis_source_commit",
                "analysis_source_tree_sha256",
                "gpu_runs_executed",
                "source_artifact",
                "boundary_policy",
            )
        }

    def _resume_or_initialize(self, manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.store.root.exists():
            self.store.initialize()
            self.store.write_json(MANIFEST_FILE, manifest)
            return None
        if not self.resume:
            raise FileExistsError(
                f"M1 boundary artifact root exists: {self.store.root}; use --resume"
            )
        if not (self.store.root / M1_CAPACITY_INTEGRITY_FILE).is_file():
            raise ValueError("existing M1 boundary artifact is not independently sealed")
        validate_m1_capacity_artifacts(self.store.root)
        existing_manifest = _read_json(self.store.root / MANIFEST_FILE)
        if self._manifest_identity(existing_manifest) != self._manifest_identity(manifest):
            raise ValueError("M1 boundary resume manifest identity mismatch")
        return _read_json(self.store.root / SUMMARY_FILE)

    @staticmethod
    def _acceptance(
        source: Mapping[str, Any],
        boundaries: CapacityBoundaryAnalysis,
    ) -> dict[str, Any]:
        execution = _mapping(source["execution"], "source execution")
        acceptance = _mapping(source["acceptance"], "source acceptance")
        checks = _mapping(source["checks"], "source acceptance checks")
        source_failures = acceptance.get("failure_reasons")
        source_failure_set = set(source_failures) if isinstance(source_failures, list) else set()
        non_knee_checks_passed = all(
            value is True for name, value in checks.items() if name != "capacity_knees_accepted"
        )
        derived_checks = {
            "source_artifact_sealed": True,
            "source_execution_complete": (
                execution.get("passed") is True
                and execution.get("planned_jobs") == 27
                and execution.get("completed_jobs") == 27
                and execution.get("failed_jobs") == 0
                and execution.get("unsafe_cleanup") is False
            ),
            "source_formal_eligible": acceptance.get("eligible") is True,
            "source_v1_failure_preserved": (
                acceptance.get("passed") is False
                and source_failure_set == _EXPECTED_SOURCE_KNEE_FAILURES
                and checks.get("capacity_knees_accepted") is False
            ),
            "source_non_knee_checks_passed": non_knee_checks_passed,
            "required_contexts_present": (
                len(boundaries.contexts) == 3
                and {context.context_tokens for context in boundaries.contexts}
                == _REQUIRED_CONTEXT_TOKENS
            ),
            "all_capacity_points_eligible": all(
                context.all_points_eligible for context in boundaries.contexts
            ),
            "slo_service_boundaries_resolved": all(
                context.slo_service_boundary.accepted for context in boundaries.contexts
            ),
            "joint_saturation_boundaries_resolved": all(
                context.joint_saturation_boundary.accepted for context in boundaries.contexts
            ),
            "numeric_thresholds_unchanged": boundaries.numeric_thresholds_modified is False,
            "boundary_analysis_accepted": boundaries.accepted,
            "gpu_reruns_not_used": True,
        }
        failures = [name for name, passed in derived_checks.items() if not passed]
        return {
            "eligible": True,
            "passed": all(derived_checks.values()),
            "checks": derived_checks,
            "failure_reasons": failures,
        }

    @staticmethod
    def _load_label(reference: object) -> str:
        if reference is None:
            return "None"
        load_id = getattr(reference, "load_id", None)
        rate = getattr(reference, "target_offered_requests_per_second", None)
        return f"{load_id} ({rate:g} rps)" if isinstance(rate, float) else str(load_id)

    def _report(
        self,
        source: Mapping[str, Any],
        boundaries: CapacityBoundaryAnalysis,
        acceptance: Mapping[str, Any],
    ) -> str:
        lines = [
            "# Long-context v5 M1 capacity boundaries v2",
            "",
            f"- Experiment: {self.experiment_id}",
            f"- Sealed source: {self.source_experiment_id}",
            f"- Source v1 accepted: {source['acceptance'].get('passed')}",
            f"- V2 boundary acceptance: {acceptance.get('passed')}",
            "- Numeric thresholds modified: false",
            "- GPU runs executed: 0",
            "",
            "## Boundaries",
            "",
            "| Context | SLO service status | Last stable | First breach | "
            "Saturation status | Pre-saturation | First joint overload |",
            "|---|---|---|---|---|---|---|",
        ]
        for context in boundaries.contexts:
            service = context.slo_service_boundary
            saturation = context.joint_saturation_boundary
            lines.append(
                "| {context} | {service_status} | {stable} | {breach} | "
                "{saturation_status} | {pre} | {overload} |".format(
                    context=context.context_id,
                    service_status=service.status,
                    stable=self._load_label(service.last_stable),
                    breach=self._load_label(service.first_slo_goodput_breach),
                    saturation_status=saturation.status,
                    pre=self._load_label(saturation.last_pre_saturation),
                    overload=self._load_label(saturation.first_joint_overload),
                )
            )
        lines.extend(
            [
                "",
                "The sealed v1 acceptance remains unchanged. This artifact separates the "
                "production SLO boundary from the mechanism-level joint saturation boundary.",
                "",
            ]
        )
        return "\n".join(lines)

    def _resume_command(self) -> str:
        return (
            "vllm-tuner longctx-m1-capacity-boundaries "
            f"--artifact-root {self.artifact_root} "
            f"--source-experiment-id {self.source_experiment_id} "
            f"--experiment-id {self.experiment_id} --resume"
        )

    def run(self) -> dict[str, Any]:
        source = self._load_source()
        boundaries = derive_capacity_boundaries(source["analysis"])
        analysis_commit, analysis_tree = self._analysis_identity()
        manifest = self._manifest(source, boundaries, analysis_commit, analysis_tree)
        resumed = self._resume_or_initialize(manifest)
        if resumed is not None:
            return resumed

        acceptance = self._acceptance(source, boundaries)
        finished_at = utc_now_iso()
        summary = {
            "schema_version": M1_CAPACITY_REANALYSIS_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M1",
            "experiment_kind": "capacity-boundary-reanalysis",
            "evidence_role": "formal-derived",
            "experiment_id": self.experiment_id,
            "finished_at": finished_at,
            "gpu_runs_executed": 0,
            "source": manifest["source_artifact"],
            "source_v1_acceptance": source["acceptance"],
            "boundary_analysis": boundaries.model_dump(mode="json"),
            "acceptance": acceptance,
            "resume": {
                "requested": self.resume,
                "root_replayed": False,
                "command": self._resume_command(),
            },
            "artifacts": {
                "root": str(self.store.root),
                "manifest": str(self.store.root / MANIFEST_FILE),
                "summary": str(self.store.root / SUMMARY_FILE),
                "report": str(self.store.root / REPORT_FILE),
                "status": str(self.store.root / STATUS_FILE),
                "integrity": str(self.store.root / M1_CAPACITY_INTEGRITY_FILE),
            },
        }
        experiment = {
            "schema_version": M1_CAPACITY_REANALYSIS_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M1",
            "experiment_kind": "capacity-boundary-reanalysis",
            "evidence_role": "formal-derived",
            "source_artifact": manifest["source_artifact"],
            "boundary_policy": manifest["boundary_policy"],
        }
        state = "accepted" if acceptance["passed"] else "completed_not_accepted"
        status = {
            "schema_version": M1_CAPACITY_REANALYSIS_SCHEMA,
            "experiment_id": self.experiment_id,
            "state": state,
            "pid": os.getpid(),
            "gpu": [],
            "log": str(self.store.root / RUNNER_LOG_FILE),
            "result": str(self.store.root),
            "eta": finished_at,
            "resume": self._resume_command(),
            "sealed": False,
            "acceptance": acceptance,
            "planned_jobs": 0,
            "completed_jobs": 0,
            "current_trial": None,
            "unsafe_cleanup": False,
            "message": "zero-GPU boundary derivation from an immutable sealed v1 source",
            "updated_at": finished_at,
        }
        self.store.write_json(EXPERIMENT_FILE, experiment)
        self.store.write_json(SUMMARY_FILE, summary)
        self.store.write_text(REPORT_FILE, self._report(source, boundaries, acceptance))
        self.store.write_json(STATUS_FILE, status)
        self.store.write_text(
            RUNNER_LOG_FILE,
            f"{finished_at} state={state} gpu_runs=0 source={self.source_experiment_id}\n",
        )
        seal_m1_capacity_artifacts(
            self.store.root,
            self.experiment_id,
            {
                "experiment_id": self.experiment_id,
                "project_line": "longctx-v5",
                "milestone": "M1",
                "experiment_kind": "capacity-boundary-reanalysis",
                "evidence_role": "formal-derived",
                "analysis_source_commit": analysis_commit,
                "source_experiment_id": self.source_experiment_id,
                "source_integrity_sha256": source["seal_sha256"],
                "source_capacity_accepted_v1": source["acceptance"].get("passed"),
                "capacity_accepted": acceptance["passed"],
                "gpu_runs_executed": 0,
            },
        )
        validate_m1_capacity_artifacts(self.store.root)
        return summary


__all__ = ["M1_CAPACITY_REANALYSIS_SCHEMA", "M1CapacityBoundaryRunner"]
