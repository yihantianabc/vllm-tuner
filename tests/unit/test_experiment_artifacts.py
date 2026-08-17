"""Artifact layout, checksum, and resume identity tests."""

import json
import subprocess

import pytest

import vllm_tuner.experiment.manifest as manifest_module
from vllm_tuner.experiment.artifacts import (
    EXPERIMENT_INTEGRITY_FILE,
    GENERATED_EXPERIMENT_FILES,
    SUMMARY_COMPACT_FILE,
    ArtifactStore,
)
from vllm_tuner.experiment.manifest import (
    build_manifest,
    sha256_file,
    sha256_json,
    validate_resume_manifest,
)
from vllm_tuner.experiment.models import (
    EnvironmentFingerprint,
    ExperimentSpec,
    TrialResult,
    TrialStatus,
    trial_provenance,
)


def _manifest(trace_hash: str = "trace") -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp",
        model="model",
        trace_sha256=trace_hash,
        workload={"name": "chat"},
        slo={"ttft_ms": 100},
        search_space={"x": [1]},
        search_space_sha256="space",
        seed=1,
        environment=EnvironmentFingerprint(python_version="3", platform="test"),
    )


def _clean_cleanup_status() -> dict[str, bool]:
    return {
        "clean": True,
        "process_group_empty": True,
        "port_available": True,
        "gpu_clean": True,
    }


def test_environment_fingerprint_tolerates_missing_sysconf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows has no os.sysconf, so physical memory remains best-effort metadata."""

    def missing_sysconf(_name: str) -> int:
        raise AttributeError("sysconf is unavailable")

    monkeypatch.setattr(manifest_module.os, "sysconf", missing_sysconf, raising=False)
    monkeypatch.setattr(manifest_module, "_run_readonly", lambda *_args, **_kwargs: None)

    fingerprint = manifest_module.collect_environment_fingerprint()

    assert fingerprint.memory_bytes is None


def _write_consistent_trial(store: ArtifactStore, trial_id: str = "trial-0") -> TrialResult:
    base = f"trials/{trial_id}"
    request = {
        "request_id": f"{trial_id}-request",
        "status": "success",
        "input_tokens": 2,
        "output_tokens": 3,
    }
    aggregate = {
        "num_requests": 1,
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 2,
        "total_output_tokens": 3,
    }
    result = TrialResult(
        trial_id=trial_id,
        **trial_provenance(trial_id, "default"),
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        client=aggregate,
        constraints={"feasible": True},
        cleanup_status=_clean_cleanup_status(),
    )
    store.write_json(f"{base}/server-command.json", {"argv": ["vllm", "serve"]})
    store.write_json(f"{base}/params.json", result.params)
    store.write_json(
        f"{base}/status.json",
        {"status": "COMPLETE", "history": [{"current": "COMPLETE"}]},
    )
    store.write_jsonl(f"{base}/request-results.jsonl", [request])
    store.write_json(
        f"{base}/benchmark-raw.json",
        {"backend": "fake", "request_results": [request], "aggregate": aggregate},
    )
    store.write_jsonl(f"{base}/prometheus.jsonl", [])
    store.write_jsonl(f"{base}/nvml.jsonl", [])
    store.write_text(f"{base}/server.log", "server completed\n")
    store.write_json(f"{base}/cleanup.json", result.cleanup_status)
    store.ensure_trial_artifacts(result)
    return result


def test_artifact_store_writes_manifest_trace_and_checksum(tmp_path) -> None:
    trace = tmp_path / "source.jsonl"
    trace.write_text('{"request_id":"a"}\n', encoding="utf-8")
    store = ArtifactStore(tmp_path / "results", "exp")
    store.initialize()
    store.save_manifest(_manifest(sha256_file(trace)))
    copied, checksum_file = store.save_trace(trace)
    assert json.loads((store.root / "manifest.json").read_text())["experiment_id"] == "exp"
    assert copied.exists()
    assert sha256_file(copied) in checksum_file.read_text()

    holdout = tmp_path / "holdout-source.jsonl"
    holdout.write_text('{"request_id":"held-out"}\n', encoding="utf-8")
    copied_holdout, holdout_checksum = store.save_holdout_trace(holdout)
    assert copied_holdout.name == "holdout-trace.jsonl"
    assert sha256_file(copied_holdout) in holdout_checksum.read_text()


def test_artifact_store_refuses_silent_reuse(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    with pytest.raises(FileExistsError):
        store.initialize()


def test_resume_manifest_detects_trace_change() -> None:
    with pytest.raises(ValueError, match="trace_sha256"):
        validate_resume_manifest(_manifest("a"), _manifest("b"))


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("constraints", {"max_peak_vram_gb": 40}),
        ("vllm_args", {"dtype": "bfloat16"}),
        (
            "environment",
            EnvironmentFingerprint(python_version="3.12", platform="test", cuda_version="12.8"),
        ),
        ("source_commit", "different-commit"),
    ],
)
def test_resume_manifest_rejects_execution_identity_changes(field, changed_value) -> None:
    existing = _manifest()
    requested = existing.model_copy(update={field: changed_value})

    with pytest.raises(ValueError, match=field):
        validate_resume_manifest(existing, requested)


def test_manifest_hashes_local_weight_shards_and_rejects_changed_resume(
    tmp_path, monkeypatch
) -> None:
    model_dir = tmp_path / "local-model"
    nested = model_dir / "nested"
    nested.mkdir(parents=True)
    shard_one = model_dir / "model-00001-of-00002.bin"
    shard_two = model_dir / "model-00002-of-00002.safetensors"
    quantized = nested / "model.GGUF"
    shard_two.write_bytes(b"second shard")
    quantized.write_bytes(b"quantized shard")
    shard_one.write_bytes(b"first shard")
    (model_dir / "optimizer.pt").write_bytes(b"not a model weight format")
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"request_id":"one"}\n', encoding="utf-8")

    fingerprint = EnvironmentFingerprint(python_version="3", platform="test")
    monkeypatch.setattr(manifest_module, "collect_environment_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(manifest_module, "git_state", lambda repository: ("commit", False, None))
    manifest_kwargs = {
        "experiment_id": "weights",
        "model": str(model_dir),
        "trace_path": trace,
        "workload": {"name": "chat"},
        "slo": {"ttft_ms": 100},
        "constraints": {},
        "gpu_config": {"device_ids": [0]},
        "telemetry": {"enabled": True},
        "study": {"seed": 1},
        "vllm_args": {},
        "search_space": {"max_num_seqs": [1, 2]},
        "seed": 1,
        "repository": tmp_path,
    }

    existing = build_manifest(**manifest_kwargs)

    assert [item.path for item in existing.model_weight_files] == [
        "model-00001-of-00002.bin",
        "model-00002-of-00002.safetensors",
        "nested/model.GGUF",
    ]
    assert [item.sha256 for item in existing.model_weight_files] == [
        sha256_file(shard_one),
        sha256_file(shard_two),
        sha256_file(quantized),
    ]
    assert existing.model_weights_sha256 == sha256_json(
        [item.model_dump(mode="json") for item in existing.model_weight_files]
    )

    shard_one.write_bytes(b"changed first shard")
    requested = build_manifest(**manifest_kwargs)
    with pytest.raises(ValueError, match="model_weight_files"):
        validate_resume_manifest(existing, requested)


def test_manifest_hashes_all_local_tokenizer_metadata(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text('{"version":"1"}\n', encoding="utf-8")
    tokenizer_config = model_dir / "tokenizer_config.json"
    tokenizer_config.write_text('{"chat_template":"A"}\n', encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"request_id":"one"}\n', encoding="utf-8")
    fingerprint = EnvironmentFingerprint(python_version="3", platform="test")
    monkeypatch.setattr(manifest_module, "collect_environment_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(manifest_module, "git_state", lambda repository: ("commit", False, None))
    kwargs = {
        "experiment_id": "tokenizer-identity",
        "model": str(model_dir),
        "trace_path": trace,
        "workload": {"name": "chat"},
        "slo": {"ttft_ms": 100},
        "constraints": {},
        "gpu_config": {"device_ids": [0]},
        "telemetry": {"enabled": True},
        "study": {"seed": 1},
        "vllm_args": {},
        "search_space": {"max_num_seqs": [1, 2]},
        "seed": 1,
        "repository": tmp_path,
    }

    existing = build_manifest(**kwargs)
    tokenizer_config.write_text(
        '{"chat_template":"B","padding_side":"left"}\n',
        encoding="utf-8",
    )
    requested = build_manifest(**kwargs)

    assert existing.tokenizer_sha256 != requested.tokenizer_sha256
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        validate_resume_manifest(existing, requested)


def test_manifest_hashes_local_generation_and_weight_index_metadata(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    generation_config = model_dir / "generation_config.json"
    generation_config.write_text('{"repetition_penalty":1.0}\n', encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"model.safetensors"}}\n',
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"request_id":"one"}\n', encoding="utf-8")
    fingerprint = EnvironmentFingerprint(python_version="3", platform="test")
    monkeypatch.setattr(manifest_module, "collect_environment_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(manifest_module, "git_state", lambda repository: ("commit", False, None))
    kwargs = {
        "experiment_id": "model-metadata-identity",
        "model": str(model_dir),
        "trace_path": trace,
        "workload": {"name": "chat"},
        "slo": {"ttft_ms": 100},
        "constraints": {},
        "gpu_config": {"device_ids": [0]},
        "telemetry": {"enabled": True},
        "study": {"seed": 1},
        "vllm_args": {},
        "search_space": {"max_num_seqs": [1, 2]},
        "seed": 1,
        "repository": tmp_path,
    }

    existing = build_manifest(**kwargs)
    generation_config.write_text('{"repetition_penalty":1.2}\n', encoding="utf-8")
    requested = build_manifest(**kwargs)

    assert existing.model_config_sha256 == requested.model_config_sha256
    assert existing.model_weights_sha256 == requested.model_weights_sha256
    assert existing.model_metadata_sha256 != requested.model_metadata_sha256
    with pytest.raises(ValueError, match="model_metadata_sha256"):
        validate_resume_manifest(existing, requested)


@pytest.mark.parametrize("tracked", [True, False], ids=["tracked", "untracked"])
def test_resume_manifest_rejects_changed_dirty_source_content(
    tmp_path, monkeypatch, tracked
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "tracked.py"
    source.write_text("value = 0\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SLOTune Test",
            "-c",
            "user.email=slotune@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    changed = source if tracked else repository / "untracked.py"
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"request_id":"one"}\n', encoding="utf-8")
    fingerprint = EnvironmentFingerprint(python_version="3", platform="test")
    monkeypatch.setattr(manifest_module, "collect_environment_fingerprint", lambda: fingerprint)
    kwargs = {
        "experiment_id": "source-identity",
        "model": "remote-model",
        "trace_path": trace,
        "workload": {"name": "chat"},
        "slo": {"ttft_ms": 100},
        "constraints": {},
        "gpu_config": {"device_ids": [0]},
        "telemetry": {"enabled": True},
        "study": {"seed": 1},
        "vllm_args": {},
        "search_space": {"max_num_seqs": [1, 2]},
        "seed": 1,
        "repository": repository,
    }

    changed.write_text("value = 1\n", encoding="utf-8")
    existing = build_manifest(**kwargs)
    changed.write_text("value = 2\n", encoding="utf-8")
    requested = build_manifest(**kwargs)

    assert existing.source_commit == requested.source_commit
    assert existing.dirty_worktree is True
    assert requested.dirty_worktree is True
    assert existing.source_tree_sha256 != requested.source_tree_sha256
    with pytest.raises(ValueError, match="source_tree_sha256"):
        validate_resume_manifest(existing, requested)


def test_save_trial_result_preserves_state_machine_history(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    trial_dir = store.trial_dir("trial-0")
    lifecycle = {"status": "STOPPING", "history": [{"to": "STOPPING"}]}
    store.write_json("trials/trial-0/status.json", lifecycle)
    result = TrialResult(
        trial_id="trial-0",
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        constraints={"feasible": True},
    )

    store.save_trial_result(result)
    store.ensure_trial_artifacts(result)

    assert json.loads((trial_dir / "status.json").read_text()) == lifecycle
    assert json.loads((trial_dir / "summary.json").read_text())["status"] == "COMPLETE"


def test_finalize_trial_layout_marks_missing_raw_evidence_unavailable(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    result = TrialResult(
        trial_id="trial-0",
        method="default",
        status=TrialStatus.FAILED,
        params={"max_num_seqs": 8},
        constraints={"feasible": False},
    )

    status = store.ensure_trial_artifacts(result)
    store.validate_trial_artifacts("trial-0", require_telemetry=True)
    with pytest.raises(ValueError, match="unavailable evidence"):
        store.validate_trial_artifacts("trial-0", require_available=True)

    assert status["complete_layout"] is True
    assert status["degraded"] is True
    assert status["files"]["prometheus.jsonl"]["data_available"] is False
    marker = json.loads(
        (store.trial_dir("trial-0") / "prometheus.jsonl").read_text().splitlines()[0]
    )
    assert marker["available"] is False
    lifecycle = json.loads((store.trial_dir("trial-0") / "status.json").read_text())
    assert lifecycle["history_available"] is False


def test_validate_available_rejects_preexisting_markers_and_empty_evidence(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    base = "trials/trial-0"
    result = TrialResult(
        trial_id="trial-0",
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        constraints={"feasible": True},
    )
    store.write_json(
        f"{base}/server-command.json",
        {"available": False, "argv": None, "environment": None},
    )
    store.write_json(f"{base}/params.json", result.params)
    store.write_json(f"{base}/status.json", {"status": "COMPLETE", "terminal": True})
    store.write_text(f"{base}/request-results.jsonl", "")
    store.write_json(
        f"{base}/benchmark-raw.json",
        {"available": False, "reason": "collector did not run"},
    )
    store.write_text(f"{base}/server.log", "")
    store.write_json(
        f"{base}/cleanup.json",
        {
            "clean": False,
            "process_group_empty": True,
            "port_available": True,
            "gpu_clean": True,
        },
    )
    store.save_trial_result(result)

    store.validate_trial_artifacts("trial-0", require_telemetry=False, require_available=False)
    with pytest.raises(ValueError, match="unavailable evidence") as error:
        store.validate_trial_artifacts("trial-0", require_telemetry=False, require_available=True)

    message = str(error.value)
    assert "server-command.json" in message
    assert "request-results.jsonl" in message
    assert "benchmark-raw.json" in message
    assert "server.log" in message
    assert "cleanup.json" in message


def test_load_trial_result_requires_a_terminal_valid_summary(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    terminal = TrialResult(
        trial_id="complete",
        method="random",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 4},
        constraints={"feasible": True},
    )
    store.save_trial_result(terminal)

    loaded = store.load_trial_result("complete")

    assert loaded == terminal
    assert store.load_trial_result("missing") is None

    nonterminal = TrialResult(
        trial_id="running",
        method="random",
        status=TrialStatus.MEASURING,
        params={"max_num_seqs": 4},
    )
    store.save_trial_result(nonterminal)
    with pytest.raises(ValueError, match="not terminal"):
        store.load_trial_result("running")


def test_trial_integrity_rejects_tampering_even_before_semantic_replay(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    result = _write_consistent_trial(store)

    store.validate_trial_integrity(result.trial_id)
    store.write_json("trials/trial-0/params.json", {"max_num_seqs": 999})

    with pytest.raises(ValueError, match="checksum mismatch: params.json"):
        store.validate_cached_trial(result, require_telemetry=False)


def test_cached_trial_rejects_cross_inconsistent_checksummed_evidence(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    result = _write_consistent_trial(store)
    store.write_json("trials/trial-0/params.json", {"max_num_seqs": 999})
    store._write_trial_integrity(result.trial_id)

    with pytest.raises(ValueError, match="inconsistent params.json"):
        store.validate_cached_trial(result, require_telemetry=False)


def test_cached_trial_rejects_raw_request_mixed_from_another_run(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    result = _write_consistent_trial(store)
    raw_path = store.trial_dir(result.trial_id) / "benchmark-raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["request_results"][0]["request_id"] = "different-run-request"
    store.write_json("trials/trial-0/benchmark-raw.json", raw)
    store._write_trial_integrity(result.trial_id)

    with pytest.raises(ValueError, match="request-results.jsonl and benchmark-raw.json"):
        store.validate_cached_trial(result, require_telemetry=False)


def test_trial_integrity_seals_dynamic_capacity_and_telemetry_files(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    result = _write_consistent_trial(store)
    base = f"trials/{result.trial_id}"
    store.write_text(f"{base}/capacity-trace.jsonl", '{"request_id":"capacity"}\n')
    store.write_text(f"{base}/capacity-trace.sha256", "trace-hash  capacity-trace.jsonl\n")
    store.write_json(
        f"{base}/capacity-point.json",
        {"offered_requests_per_sec": 4.0, "repeat": 0, "trace_sha256": "trace-hash"},
    )
    store.write_jsonl(f"{base}/telemetry.jsonl", [{"sample": 1}])
    store.write_jsonl(
        f"{base}/scheduler-decisions.jsonl",
        [{"step_index": 0, "controller_state": "DISABLED"}],
    )

    store.seal_trial_artifacts(result)
    integrity = json.loads(
        (store.trial_dir(result.trial_id) / "artifact-integrity.json").read_text()
    )

    assert {
        "capacity-trace.jsonl",
        "capacity-trace.sha256",
        "capacity-point.json",
        "telemetry.jsonl",
        "scheduler-decisions.jsonl",
    }.issubset(integrity["files"])
    summary = store.load_trial_result(result.trial_id)
    assert summary is not None
    assert summary.artifacts["scheduler-decisions.jsonl"].endswith("/scheduler-decisions.jsonl")
    store.validate_trial_integrity(result.trial_id)

    store.write_text(f"{base}/capacity-trace.jsonl", '{"request_id":"tampered"}\n')
    with pytest.raises(ValueError, match="checksum mismatch: capacity-trace.jsonl"):
        store.validate_trial_integrity(result.trial_id)


def test_cached_trial_round_trips_one_failed_request_out_of_five_hundred(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    trial_id = "trial-500"
    base = f"trials/{trial_id}"
    requests = [
        {
            "request_id": f"request-{index}",
            "status": "success",
            "input_tokens": 2,
            "output_tokens": 3,
        }
        for index in range(499)
    ]
    requests.append(
        {
            "request_id": "request-499",
            "status": "failed",
            "input_tokens": 17,
            "output_tokens": 5,
            "error_type": "http_500",
        }
    )
    aggregate = {
        "num_requests": 500,
        "completed": 499,
        "failed": 1,
        "total_input_tokens": 998,
        "total_output_tokens": 1497,
    }
    result = TrialResult(
        trial_id=trial_id,
        method="default",
        status=TrialStatus.COMPLETE,
        params={"max_num_seqs": 8},
        client=aggregate,
        constraints={"feasible": True},
        cleanup_status=_clean_cleanup_status(),
    )
    store.write_json(f"{base}/server-command.json", {"argv": ["vllm", "serve"]})
    store.write_json(f"{base}/params.json", result.params)
    store.write_json(
        f"{base}/status.json",
        {"status": "COMPLETE", "history": [{"current": "COMPLETE"}]},
    )
    store.write_jsonl(f"{base}/request-results.jsonl", requests)
    store.write_json(
        f"{base}/benchmark-raw.json",
        {"backend": "fake", "request_results": requests, "aggregate": aggregate},
    )
    store.write_jsonl(f"{base}/prometheus.jsonl", [])
    store.write_jsonl(f"{base}/nvml.jsonl", [])
    store.write_text(f"{base}/server.log", "server completed\n")
    store.write_json(f"{base}/cleanup.json", result.cleanup_status)

    store.ensure_trial_artifacts(result)
    loaded = store.load_trial_result(trial_id)

    assert loaded is not None
    assert loaded.client["completed"] == 499
    assert loaded.client["failed"] == 1
    assert loaded.client["total_input_tokens"] == 998
    assert loaded.client["total_output_tokens"] == 1497
    store.validate_cached_trial(loaded, require_telemetry=False)


def _write_attestable_experiment(store: ArtifactStore) -> TrialResult:
    store.initialize()
    trace_path = store.write_text("trace.jsonl", '{"request_id":"search"}\n')
    holdout_path = store.write_text("holdout-trace.jsonl", '{"request_id":"holdout"}\n')
    trace_sha256 = sha256_file(trace_path)
    holdout_sha256 = sha256_file(holdout_path)
    manifest = _manifest(trace_sha256).model_copy(
        update={
            "holdout_trace_sha256": holdout_sha256,
            "source_commit": "measurement-commit",
            "source_tree_sha256": "measurement-tree",
            "dirty_worktree": False,
            "experiment_config_sha256": "experiment-config",
        }
    )
    store.save_manifest(manifest)
    store.write_yaml("experiment.yaml", {"model": "model"})
    store.write_text("trace.sha256", f"{trace_sha256}  trace.jsonl\n")
    store.write_text("holdout-trace.sha256", f"{holdout_sha256}  holdout-trace.jsonl\n")
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
        "requests": [{"request_id": "scheduler-request"}],
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
    store.write_json(
        "summary.json",
        {
            "experiment_id": store.root.name,
            "manifest": manifest.model_dump(mode="json"),
            "scheduler_ablation": scheduler,
        },
    )
    store.write_text("report/report.md", "# report\n")
    store.write_text("report/report.html", "<html></html>\n")
    store.write_json("report/plot-manifest.json", {"schema_version": 1, "plots": {}})
    return _write_consistent_trial(store, "default-0000")


def test_experiment_attestation_compacts_scheduler_and_seals_exact_tree(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    result = _write_attestable_experiment(store)
    raw_path = store.aggregate_dir / "scheduler-ablation.json"
    raw_sha256 = sha256_file(raw_path)
    anchor = store.trial_dir(result.trial_id) / "artifact-integrity.json"
    anchor_sha256 = sha256_file(anchor)
    summary_path = store.root / "summary.json"
    summary_sha256 = sha256_file(summary_path)
    summary_size = summary_path.stat().st_size

    attested = store.attest_experiment_artifacts(
        attestation={
            "kind": "unit-test-attestation",
            "attestation_source_commit": "tool-commit",
            "attestation_source_tree_sha256": "tool-tree",
            "attestation_dirty_worktree": True,
            "measurement_source_commit": "caller-must-not-override",
            "measurement_source_tree_sha256": "caller-must-not-override",
            "trace_sha256": "caller-must-not-override",
        }
    )

    assert attested["already_sealed"] is False
    store.validate_experiment_integrity()
    assert (store.root / EXPERIMENT_INTEGRITY_FILE).is_file()
    assert sha256_file(raw_path) == raw_sha256
    assert sha256_file(anchor) == anchor_sha256
    assert sha256_file(summary_path) == summary_sha256
    assert summary_path.stat().st_size == summary_size
    raw_summary = json.loads(summary_path.read_text())
    assert raw_summary["scheduler_ablation"] == json.loads(raw_path.read_text())
    compact_summary = json.loads((store.root / SUMMARY_COMPACT_FILE).read_text())
    compact = compact_summary["scheduler_ablation"]
    assert compact["raw_artifact"] == "aggregate/scheduler-ablation.json"
    assert compact["raw_sha256"] == raw_sha256
    assert compact["raw_size_bytes"] == raw_path.stat().st_size
    assert "requests" not in compact["calibration"]["adaptive"]
    assert json.loads(raw_path.read_text())["calibration"]["adaptive"]["requests"]
    original = compact_summary["experiment_attestation"]["original_summary"]
    assert original == {
        "path": "summary.json",
        "size_bytes": summary_size,
        "sha256": summary_sha256,
    }
    audit = json.loads((store.root / "experiment-audit.json").read_text())
    assert audit["trial_semantic_validated"] == 1
    assert audit["legacy_provenance_trial_count"] == 0
    lineage = json.loads((store.root / "lineage.json").read_text())
    assert lineage["trials"][0]["provenance_kind"] == "recorded"
    integrity = json.loads((store.root / EXPERIMENT_INTEGRITY_FILE).read_text())
    record = integrity["attestations"][-1]
    assert record["attestation_kind"] == "unit-test-attestation"
    assert record["measurement_source_commit"] == "measurement-commit"
    assert record["measurement_source_tree_sha256"] == "measurement-tree"
    assert record["measurement_dirty_worktree"] is False
    assert record["attestation_source_commit"] == "tool-commit"
    assert record["attestation_source_tree_sha256"] == "tool-tree"
    assert record["attestation_dirty_worktree"] is True
    assert record["attested_at_utc"]
    assert record["experiment_config_sha256"] == "experiment-config"
    assert record["trace_sha256"] == raw_summary["manifest"]["trace_sha256"]
    assert SUMMARY_COMPACT_FILE in integrity["files"]

    first_seal = sha256_file(store.root / EXPERIMENT_INTEGRITY_FILE)
    repeated = store.attest_experiment_artifacts(attestation={"kind": "unit-test-attestation"})
    assert repeated["already_sealed"] is True
    assert sha256_file(store.root / EXPERIMENT_INTEGRITY_FILE) == first_seal

    resealed = store.attest_experiment_artifacts(
        attestation={"kind": "authorized-reseal"},
        reseal=True,
    )
    assert resealed["already_sealed"] is False
    assert sha256_file(store.root / EXPERIMENT_INTEGRITY_FILE) != first_seal
    assert sha256_file(raw_path) == raw_sha256
    assert sha256_file(anchor) == anchor_sha256
    assert sha256_file(summary_path) == summary_sha256
    assert summary_path.stat().st_size == summary_size
    store.validate_experiment_integrity()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("aggregate/scheduler-ablation.json", "artifact checksum mismatch"),
        ("trials/default-0000/server.log", "artifact checksum mismatch"),
        ("unsealed-root.txt", "file set mismatch"),
    ],
)
def test_experiment_integrity_rejects_root_child_and_added_file_tampering(
    tmp_path, target, expected
) -> None:
    store = ArtifactStore(tmp_path, "exp")
    _write_attestable_experiment(store)
    store.attest_experiment_artifacts(attestation={"kind": "initial"})

    store.write_text(target, "tampered\n")

    with pytest.raises(ValueError, match=expected):
        store.validate_experiment_integrity()
    with pytest.raises(ValueError, match=expected):
        store.attest_experiment_artifacts(
            attestation={"kind": "must-not-bless-corruption"},
            reseal=True,
        )


def test_failed_first_attestation_does_not_write_partial_root_views(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    _write_attestable_experiment(store)
    summary_path = store.root / "summary.json"
    summary_sha256 = sha256_file(summary_path)
    store.write_text("trials/default-0000/server.log", "corrupted before attestation\n")

    with pytest.raises(ValueError, match="artifact checksum mismatch: server.log"):
        store.attest_experiment_artifacts(attestation={"kind": "must-fail"})

    assert sha256_file(summary_path) == summary_sha256
    for relative in (
        EXPERIMENT_INTEGRITY_FILE,
        *GENERATED_EXPERIMENT_FILES,
    ):
        assert not (store.root / relative).exists()


def test_missing_required_input_fails_before_writing_any_attestation_view(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    _write_attestable_experiment(store)
    summary_path = store.root / "summary.json"
    summary_sha256 = sha256_file(summary_path)
    summary_size = summary_path.stat().st_size
    (store.root / "report/report.html").unlink()

    with pytest.raises(ValueError, match=r"input preflight failed.*report/report.html"):
        store.attest_experiment_artifacts(attestation={"kind": "must-fail"})

    assert sha256_file(summary_path) == summary_sha256
    assert summary_path.stat().st_size == summary_size
    assert not (store.root / EXPERIMENT_INTEGRITY_FILE).exists()
    for relative in GENERATED_EXPERIMENT_FILES:
        assert not (store.root / relative).exists()


def test_legacy_repeat_and_holdout_lineage_is_derived_without_rewriting_children(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path, "exp")
    _write_attestable_experiment(store)
    manifest_path = store.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_schema_version"] = "4"
    store.write_json("manifest.json", manifest)
    summary = json.loads((store.root / "summary.json").read_text())
    summary["manifest"] = manifest
    store.write_json("summary.json", summary)

    child_hashes: dict[str, str] = {}
    for phase in ("repeat", "holdout"):
        trial_id = f"{phase}-default-0-0"
        result = _write_consistent_trial(store, trial_id).model_copy(
            update={
                "method": phase,
                "phase": None,
                "source_method": None,
                "source_trial_id": None,
            }
        )
        store.seal_trial_artifacts(result)
        child_hashes[trial_id] = sha256_file(store.trial_dir(trial_id) / "artifact-integrity.json")

    store.attest_experiment_artifacts(attestation={"kind": "legacy-lineage"})

    lineage = json.loads((store.root / "lineage.json").read_text())
    trials = {row["trial_id"]: row for row in lineage["trials"]}
    for phase in ("repeat", "holdout"):
        trial_id = f"{phase}-default-0-0"
        assert trials[trial_id] == {
            "trial_id": trial_id,
            "recorded_method": phase,
            "phase": phase,
            "source_method": "default",
            "source_trial_id": "default-0000",
            "provenance_kind": "derived_from_trial_id",
            "path": f"trials/{trial_id}/artifact-integrity.json",
            "size_bytes": (store.trial_dir(trial_id) / "artifact-integrity.json").stat().st_size,
            "sha256": child_hashes[trial_id],
        }
        assert (
            sha256_file(store.trial_dir(trial_id) / "artifact-integrity.json")
            == child_hashes[trial_id]
        )
    audit = json.loads((store.root / "experiment-audit.json").read_text())
    assert audit["derived_repeat_holdout_trial_count"] == 2
    assert audit["derived_provenance_trial_count"] == 2


def test_attestation_derives_legacy_capacity_empirical_rates_from_sealed_trials(
    tmp_path,
) -> None:
    store = ArtifactStore(tmp_path, "exp")
    _write_attestable_experiment(store)
    trial_id = "capacity-rate-1-repeat-0"
    result = _write_consistent_trial(store, trial_id)
    client = {
        **result.client,
        "offered_requests_per_sec": 1.0,
        "empirical_scheduled_requests_per_sec": 0.8,
        "achieved_requests_per_sec": 0.75,
    }
    result = result.model_copy(update={"client": client})
    benchmark_path = store.trial_dir(trial_id) / "benchmark-raw.json"
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["aggregate"] = client
    store.write_json(f"trials/{trial_id}/benchmark-raw.json", benchmark)
    store.seal_trial_artifacts(result)
    summary_path = store.root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["capacity_sweep"] = {
        "points": [
            {
                "trial_id": trial_id,
                "offered_requests_per_sec": 1.0,
                "measured_offered_requests_per_sec": 1.0,
                "achieved_requests_per_sec": 0.75,
            }
        ],
        "by_rate": [],
    }
    store.write_json("summary.json", summary)
    summary_sha256 = sha256_file(summary_path)

    store.attest_experiment_artifacts(attestation={"kind": "legacy-capacity"})

    assert sha256_file(summary_path) == summary_sha256
    audit = json.loads((store.root / "experiment-audit.json").read_text())
    semantics = audit["capacity_rate_semantics"]
    assert semantics["summary_capacity_schema"] == "legacy_target_alias_v1"
    assert semantics["trials"] == [
        {
            "trial_id": trial_id,
            "status": "COMPLETE",
            "target_offered_requests_per_sec": 1.0,
            "empirical_scheduled_requests_per_sec": 0.8,
            "achieved_requests_per_sec": 0.75,
        }
    ]
    assert semantics["by_target_rate"] == [
        {
            "target_offered_requests_per_sec": 1.0,
            "measured_count": 1,
            "median_empirical_scheduled_requests_per_sec": 0.8,
            "min_empirical_scheduled_requests_per_sec": 0.8,
            "max_empirical_scheduled_requests_per_sec": 0.8,
        }
    ]
    compact = json.loads((store.root / SUMMARY_COMPACT_FILE).read_text())
    assert compact["capacity_rate_semantics"] == semantics


def test_capacity_attestation_semantics_rejects_point_and_trial_mismatch(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    result = TrialResult(
        trial_id="capacity-rate-1-repeat-0",
        **trial_provenance("capacity-rate-1-repeat-0", "capacity"),
        status=TrialStatus.COMPLETE,
        params={},
        client={
            "offered_requests_per_sec": 1.0,
            "empirical_scheduled_requests_per_sec": 0.8,
        },
    )
    mismatched = {
        "capacity_sweep": {
            "points": [
                {
                    "trial_id": result.trial_id,
                    "target_offered_requests_per_sec": 1.0,
                    "empirical_scheduled_requests_per_sec": 0.9,
                }
            ]
        }
    }

    with pytest.raises(ValueError, match="inconsistent empirical scheduled rate"):
        store._capacity_rate_semantics(mismatched, [result])

    with pytest.raises(ValueError, match="Capacity summary/trial set mismatch"):
        store._capacity_rate_semantics({"capacity_sweep": {"points": []}}, [result])


def test_schema_five_cached_repeat_requires_canonical_provenance(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    store.save_manifest(_manifest())
    result = _write_consistent_trial(store, "repeat-default-0-0")

    store.validate_cached_trial(result, require_telemetry=False)
    result.source_method = "random"
    store.seal_trial_artifacts(result)

    with pytest.raises(ValueError, match="provenance source_method"):
        store.validate_cached_trial(result, require_telemetry=False)
