"""Frozen M4 runtime matrix for adaptive Prefill calibration and held-out traces."""

from __future__ import annotations

import html
import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from vllm_tuner.analysis.nonstationary import (
    aggregate_policy_trials,
    summarize_labeled_requests,
)
from vllm_tuner.config.models import TuningConfig
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.workloads.trace import WorkloadTrace

from .artifacts import EXPERIMENT_INTEGRITY_FILE
from .manifest import git_state, sha256_file, source_tree_sha256
from .models import TrialResult
from .runner import SLOTuneExperimentRunner

TRACE_KINDS = ("calibration", "heldout")


@dataclass(frozen=True)
class FormalJob:
    """One policy/load/trace repetition in randomized execution order."""

    load: str
    policy: str
    trace_kind: str
    repeat: int

    @property
    def trial_id(self) -> str:
        return f"formal-{self.trace_kind}-repeat-{self.repeat}"


@dataclass
class FormalContext:
    """One independently sealed policy/load experiment."""

    load: str
    policy: str
    runner: SLOTuneExperimentRunner
    calibration: WorkloadTrace
    heldout: WorkloadTrace
    params: dict[str, Any]
    sealed: bool

    def trace(self, trace_kind: str) -> WorkloadTrace:
        return self.calibration if trace_kind == "calibration" else self.heldout


def load_formal_protocol(path: str | Path) -> dict[str, Any]:
    """Load the frozen matrix and reject missing structural fields."""
    protocol_path = Path(path).expanduser().resolve()
    value = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Formal protocol must be a YAML object")
    required = {
        "pilot_model",
        "common_server",
        "policies",
        "load_points",
        "slo_tiers_ms",
        "primary_slo_tier",
        "formal_execution",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Formal protocol is missing fields: {missing}")
    if value["primary_slo_tier"] not in value["slo_tiers_ms"]:
        raise ValueError("primary_slo_tier is not defined")
    execution = value["formal_execution"]
    if execution.get("freeze_policy_parameters") is not True:
        raise ValueError("Formal protocol must freeze policy parameters")
    if execution.get("policy_order_randomized_per_repeat") is not True:
        raise ValueError("Formal protocol must randomize policy order per repeat")
    if int(execution.get("repeats", 0)) < 3:
        raise ValueError("Formal protocol requires at least three repetitions")
    return value


def formal_experiment_id(prefix: str, load: str, policy: str) -> str:
    return f"{prefix}-{load}-{policy.replace('_', '-')}"


def build_formal_config(
    protocol: Mapping[str, Any],
    *,
    load: str,
    policy: str,
    resume: bool,
) -> TuningConfig:
    """Translate one frozen matrix cell into a runtime TuningConfig."""
    common = dict(protocol["common_server"])
    load_point = dict(protocol["load_points"][load])
    execution = dict(protocol["formal_execution"])
    policies = protocol["policies"]
    policy_spec = dict(policies[policy])
    primary = protocol["slo_tiers_ms"][protocol["primary_slo_tier"]]

    adaptive = dict(policies["adaptive"]["adaptive_prefill"])
    if policy == "stock":
        adaptive = {"enabled": False}
    else:
        adaptive.update(policy_spec["adaptive_prefill"])

    vllm_args: dict[str, Any] = {
        "max-model-len": int(common["max_model_len"]),
        "no-async-scheduling": not bool(common["async_scheduling"]),
    }
    scheduler_cls = policy_spec.get("scheduler_cls")
    if scheduler_cls:
        vllm_args["scheduler-cls"] = scheduler_cls

    config = {
        "model": protocol["pilot_model"],
        "gpu": {"device_ids": [0], "count": 1},
        "slo": {
            "ttft_ms": primary["ttft"],
            "tpot_ms": primary["tpot"],
            "e2e_ms": primary["e2e"],
        },
        "constraints": {
            "max_error_rate": 0.0,
            "max_peak_vram_mb": 31000,
            "max_memory_utilization": 0.95,
            "require_no_oom": True,
            "require_server_alive": True,
        },
        "search_space": {
            "gpu_memory_utilization": [common["gpu_memory_utilization"]] * 2,
            "max_num_batched_tokens": [common["max_num_batched_tokens"]],
            "max_num_seqs": [common["max_num_seqs"]],
            "tensor_parallel_size": common["tensor_parallel_size"],
            "pipeline_parallel_size": common["pipeline_parallel_size"],
        },
        "workload": {
            "name": f"nonstationary-formal-{load}",
            "dataset_name": "frozen:nonstationary-formal",
            "sample_size": execution["measured_requests_per_trace"],
            "prompt_length_distribution": "weighted",
            "warmup_requests": execution["warmup_requests"],
            "max_concurrency": execution["max_concurrency"],
            "request_rate": load_point["empirical_requests_per_second"],
            "max_tokens": 256,
            "ignore_eos": execution["ignore_eos"],
            "seed": 2026,
            "request_timeout_seconds": 300,
            "benchmark_backend": "sse",
        },
        "telemetry": {
            "enabled": True,
            "interval_ms": 100,
            "collect_nvml": True,
        },
        "baseline": {"enabled": False},
        "study": {
            "trial_budget": 1,
            "timeout_minutes": 240,
            "prune_enabled": False,
            "n_startup_trials": 1,
            "methods": ["random"],
            "repeat_count": execution["repeats"],
            "top_candidates": 1,
            "holdout_enabled": True,
            "resume": resume,
        },
        "adaptive_prefill": adaptive,
        "vllm_args": vllm_args,
    }
    return TuningConfig.model_validate(config)


def formal_job_order(
    protocol: Mapping[str, Any],
    *,
    seed: int = 2026,
    loads: Sequence[str] | None = None,
    policies: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
    trace_kinds: Sequence[str] | None = None,
) -> list[FormalJob]:
    """Randomize policy order independently inside every repeat/trace/load block."""
    selected_loads = list(loads or protocol["load_points"])
    selected_policies = list(policies or protocol["policies"])
    selected_repeats = list(repeats or range(int(protocol["formal_execution"]["repeats"])))
    selected_traces = list(trace_kinds or TRACE_KINDS)
    if any(trace_kind not in TRACE_KINDS for trace_kind in selected_traces):
        raise ValueError(f"trace kinds must be drawn from {TRACE_KINDS}")
    jobs: list[FormalJob] = []
    for repeat in selected_repeats:
        for load_index, load in enumerate(selected_loads):
            for trace_index, trace_kind in enumerate(selected_traces):
                policy_order = list(selected_policies)
                block_seed = seed + repeat * 1009 + load_index * 97 + trace_index * 17
                random.Random(block_seed).shuffle(policy_order)
                jobs.extend(FormalJob(load, policy, trace_kind, repeat) for policy in policy_order)
    return jobs


def _read_request_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * percentile / 100))
    return ordered[index]


def _decision_summary(context: FormalContext, trial_ids: Sequence[str]) -> dict[str, Any]:
    states: Counter[str] = Counter()
    cpu_times: list[float] = []
    steps = 0
    starvation_violations = 0
    for trial_id in trial_ids:
        path = context.runner.artifacts.trials_dir / trial_id / "scheduler-decisions.jsonl"
        if not path.is_file():
            continue
        for row in _read_request_rows(path):
            if row.get("available") is False:
                continue
            steps += 1
            states[str(row.get("controller_state", "UNKNOWN"))] += 1
            value = row.get("scheduler_cpu_time_us")
            if isinstance(value, (int, float)):
                cpu_times.append(float(value))
            if "max_wait_progress_not_met" in str(row.get("reason_code", "")):
                starvation_violations += 1
    return {
        "steps": steps,
        "state_counts": dict(sorted(states.items())),
        "scheduler_cpu_time_us_p50": statistics.median(cpu_times) if cpu_times else None,
        "scheduler_cpu_time_us_p99": _percentile(cpu_times, 99),
        "max_wait_progress_not_met_steps": starvation_violations,
    }


def _trial_row(result: TrialResult) -> dict[str, Any]:
    return {
        "trial_id": result.trial_id,
        "phase": result.phase,
        "status": result.status.value,
        "feasible": result.constraints.get("feasible", False),
        "completed": result.client.get("completed"),
        "failed": result.client.get("failed"),
        "goodput_requests_per_sec": result.client.get("goodput_requests_per_sec"),
        "achieved_requests_per_sec": result.client.get("achieved_requests_per_sec"),
        "p99_ttft_ms": result.client.get("p99_ttft_ms"),
        "p99_tpot_ms": result.client.get("p99_tpot_ms"),
        "p99_e2e_ms": result.client.get("p99_e2e_ms"),
        "peak_memory_mb": result.gpu.get("peak_memory_mb"),
        "cleanup_clean": (result.cleanup_status or {}).get("clean"),
    }


class AdaptivePrefillFormalMatrix:
    """Execute and seal the exact frozen matrix without a search observation."""

    def __init__(
        self,
        protocol_path: str | Path,
        *,
        results_root: str | Path,
        experiment_prefix: str,
        repository: str | Path,
        tokenizer: Any | None = None,
    ) -> None:
        self.protocol_path = Path(protocol_path).expanduser().resolve()
        self.protocol = load_formal_protocol(self.protocol_path)
        self.results_root = Path(results_root).expanduser().resolve()
        self.experiment_prefix = experiment_prefix
        self.repository = Path(repository).resolve()
        self.tokenizer = tokenizer
        self.contexts: dict[tuple[str, str], FormalContext] = {}

    def _trace_path(self, load: str, trace_kind: str) -> Path:
        load_point = self.protocol["load_points"][load]
        key = "calibration_trace" if trace_kind == "calibration" else "heldout_trace"
        checksum_key = "calibration_sha256" if trace_kind == "calibration" else "heldout_sha256"
        path = (self.protocol_path.parent / load_point[key]).resolve()
        expected = load_point[checksum_key]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Frozen {load}/{trace_kind} trace checksum mismatch: "
                f"expected {expected}, found {actual}"
            )
        return path

    def _prepare_context(self, load: str, policy: str) -> FormalContext:
        key = (load, policy)
        if key in self.contexts:
            return self.contexts[key]
        experiment_id = formal_experiment_id(self.experiment_prefix, load, policy)
        root = self.results_root / experiment_id
        resume = root.exists()
        config = build_formal_config(
            self.protocol,
            load=load,
            policy=policy,
            resume=resume,
        )
        calibration_path = self._trace_path(load, "calibration")
        heldout_path = self._trace_path(load, "heldout")
        runner = SLOTuneExperimentRunner(
            config,
            experiment_id,
            results_root=self.results_root,
            repository=self.repository,
            trace_path=calibration_path,
            holdout_trace_path=heldout_path,
            tokenizer=self.tokenizer,
            require_clean_source=True,
        )
        calibration = WorkloadTrace.read(
            calibration_path,
            seed=2026,
            profile=f"nonstationary-formal-{load}",
            request_rate=config.workload.request_rate,
            burstiness=1.0,
        )
        heldout = WorkloadTrace.read(
            heldout_path,
            seed=3035,
            profile=f"nonstationary-formal-{load}-heldout",
            request_rate=config.workload.request_rate,
            burstiness=1.0,
        )
        frozen_calibration = runner._trace_file(calibration, "search")
        frozen_heldout = runner._trace_file(heldout, "holdout")
        runner._initialize_artifacts(frozen_calibration, frozen_heldout)
        params = {
            "gpu_memory_utilization": config.search_space.gpu_memory_utilization[0],
            "max_num_batched_tokens": config.search_space.max_num_batched_tokens[0],
            "max_num_seqs": config.search_space.max_num_seqs[0],
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        }
        context = FormalContext(
            load=load,
            policy=policy,
            runner=runner,
            calibration=calibration,
            heldout=heldout,
            params=params,
            sealed=(root / EXPERIMENT_INTEGRITY_FILE).is_file(),
        )
        self.contexts[key] = context
        return context

    async def run_job(self, job: FormalJob) -> TrialResult:
        context = self._prepare_context(job.load, job.policy)
        cached = context.runner._load_cached_trial(context.params, job.trial_id)
        if cached is not None:
            return cached
        if context.sealed:
            raise ValueError(f"sealed experiment is missing trial {job.trial_id}")
        controller = TrialController(
            context.runner.config,
            context.trace(job.trace_kind),
            context.runner.artifacts,
            tokenizer=self.tokenizer,
        )
        result = await controller.run_trial(context.params, job.trial_id, job.policy)
        result.method = job.policy
        result.phase = "held_out" if job.trace_kind == "heldout" else "calibration"
        result.source_method = job.policy
        result.source_trial_id = None
        context.runner._finalize_trial_artifacts(result)
        return result

    def _expected_trial_ids(self) -> list[str]:
        repeats = int(self.protocol["formal_execution"]["repeats"])
        return [
            f"formal-{trace_kind}-repeat-{repeat}"
            for trace_kind in TRACE_KINDS
            for repeat in range(repeats)
        ]

    def finalize_context(self, context: FormalContext) -> bool:
        """Write phase views and seal a context only after all six trials exist."""
        if context.sealed:
            context.runner.artifacts.validate_experiment_integrity()
            return True
        expected = self._expected_trial_ids()
        results = [context.runner.artifacts.load_trial_result(name) for name in expected]
        if any(result is None for result in results):
            return False
        complete_results = [result for result in results if result is not None]
        by_phase = {
            "calibration": [result for result in complete_results if result.phase == "calibration"],
            "held_out": [result for result in complete_results if result.phase == "held_out"],
        }
        analyses: dict[str, Any] = {}
        for tier, thresholds in self.protocol["slo_tiers_ms"].items():
            slo = {
                "ttft_ms": float(thresholds["ttft"]),
                "tpot_ms": float(thresholds["tpot"]),
                "e2e_ms": float(thresholds["e2e"]),
            }
            tier_analysis: dict[str, Any] = {}
            for phase, phase_results in by_phase.items():
                trace = context.calibration if phase == "calibration" else context.heldout
                trial_summaries: list[dict[str, Any]] = []
                unavailable_trials: list[dict[str, str]] = []
                for result in phase_results:
                    try:
                        trial_summaries.append(
                            summarize_labeled_requests(
                                trace,
                                _read_request_rows(
                                    context.runner.artifacts.trials_dir
                                    / result.trial_id
                                    / "request-results.jsonl"
                                ),
                                slo=slo,
                            )
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        unavailable_trials.append(
                            {
                                "trial_id": result.trial_id,
                                "reason": f"{type(error).__name__}: {error}",
                            }
                        )
                aggregate = (
                    aggregate_policy_trials(trial_summaries)
                    if trial_summaries
                    else {"trials": 0, "overall": {}, "phases": {}}
                )
                aggregate.update(
                    {
                        "expected_trials": len(phase_results),
                        "analyzed_trials": len(trial_summaries),
                        "complete_evidence": not unavailable_trials,
                        "unavailable_trials": unavailable_trials,
                    }
                )
                tier_analysis[phase] = aggregate
            analyses[tier] = tier_analysis
            context.runner.artifacts.write_json(
                f"aggregate/formal-phase-analysis-{tier}.json", tier_analysis
            )

        decision_summary = {
            phase: _decision_summary(
                context,
                [result.trial_id for result in phase_results],
            )
            for phase, phase_results in by_phase.items()
        }
        primary_tier = self.protocol["primary_slo_tier"]
        scheduler = {
            "schema_version": 1,
            "kind": "runtime_formal_policy_evidence",
            "policy": context.policy,
            "load": context.load,
            "has_negative_result": False,
            "negative_gain_conditions": [],
            "primary_slo_tier": primary_tier,
            "runtime_phase_analysis": analyses[primary_tier],
            "scheduler_decisions": decision_summary,
        }
        context.runner.artifacts.write_json("aggregate/scheduler-ablation.json", scheduler)
        trial_rows = [_trial_row(result) for result in complete_results]
        context.runner.artifacts.write_jsonl("aggregate/formal-trials.jsonl", trial_rows)

        report_lines = [
            f"# M4 Formal runtime: {context.policy} / {context.load}",
            "",
            "This report contains real vLLM runtime evidence, not simulator output.",
            "",
            "| Trial | Trace | Status | Completed | Failed | Achieved req/s | p99 TTFT | p99 TPOT |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in trial_rows:
            report_lines.append(
                "| {trial_id} | {phase} | {status} | {completed} | {failed} | "
                "{achieved_requests_per_sec} | {p99_ttft_ms} | {p99_tpot_ms} |".format(**row)
            )
        report_lines.extend(
            [
                "",
                "All strict/medium/loose phase analyses are stored under `aggregate/`.",
                "Cross-policy comparisons and the offline Oracle are produced only after every ",
                "policy cell is sealed.",
            ]
        )
        markdown = "\n".join(report_lines) + "\n"
        context.runner.artifacts.write_text("report/report.md", markdown)
        context.runner.artifacts.write_text(
            "report/report.html",
            "<!doctype html><html><body><pre>" + html.escape(markdown) + "</pre></body></html>\n",
        )
        context.runner.artifacts.write_json(
            "report/plot-manifest.json",
            {
                "schema_version": 1,
                "plots": {},
                "reason": "Cross-policy plots are emitted by the completed matrix audit",
            },
        )
        manifest = context.runner.manifest
        assert manifest is not None
        manifest.report_artifacts = {
            "files": {
                "markdown": "report/report.md",
                "html": "report/report.html",
                "plot_manifest": "report/plot-manifest.json",
            },
            "plots": {},
        }
        context.runner.artifacts.save_manifest(manifest)
        summary = {
            "experiment_id": manifest.experiment_id,
            "manifest": manifest.model_dump(mode="json"),
            "formal_policy": context.policy,
            "formal_load": context.load,
            "trials": trial_rows,
            "repetitions": [row for row in trial_rows if row["phase"] == "calibration"],
            "holdout": [row for row in trial_rows if row["phase"] == "held_out"],
            "capacity_sweep": {"points": [], "by_rate": []},
            "slo_tier_analysis": analyses,
            "scheduler_ablation": scheduler,
            "report": {
                "markdown": str(context.runner.artifacts.root / "report/report.md"),
                "html": str(context.runner.artifacts.root / "report/report.html"),
            },
        }
        context.runner.artifacts.write_json("summary.json", summary)
        commit, dirty, _ = git_state(self.repository)
        if dirty:
            raise ValueError("Formal results cannot be sealed from a dirty source tree")
        context.runner.artifacts.attest_experiment_artifacts(
            attestation={
                "kind": "adaptive-prefill-formal-runtime",
                "attestation_source_commit": commit,
                "attestation_source_tree_sha256": source_tree_sha256(self.repository),
                "attestation_dirty_worktree": dirty,
            }
        )
        context.sealed = True
        return True

    def finalize_ready_contexts(self) -> list[Path]:
        sealed: list[Path] = []
        for context in self.contexts.values():
            if self.finalize_context(context):
                sealed.append(context.runner.artifacts.root)
        return sealed
