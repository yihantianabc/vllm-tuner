"""Artifact layout, checksum, and resume identity tests."""

import json
import subprocess

import pytest

import vllm_tuner.experiment.manifest as manifest_module
from vllm_tuner.experiment.artifacts import ArtifactStore
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

    store.seal_trial_artifacts(result)
    integrity = json.loads(
        (store.trial_dir(result.trial_id) / "artifact-integrity.json").read_text()
    )

    assert {
        "capacity-trace.jsonl",
        "capacity-trace.sha256",
        "capacity-point.json",
        "telemetry.jsonl",
    }.issubset(integrity["files"])
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
