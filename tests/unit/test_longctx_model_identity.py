"""Tests for fail-closed formal-model file and identity verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from vllm_tuner.longctx.model_identity import (
    ModelFileStatus,
    ModelIdentityError,
    ModelIdentityFacts,
    inspect_model_identity,
    load_model_lock,
    require_model_identity,
)

REPOSITORY_ID = "Qwen/Test-7B-Instruct"
REVISION = "0123456789abcdef0123456789abcdef01234567"
PARAMETER_COUNT = 7_615_616_512


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _model_dir(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payloads = {
        "config.json": b'{"model_type":"test"}\n',
        "tokenizer.json": b'{"tokenizer":"ok"}\n',
        "model-00001-of-00001.safetensors": b"small deterministic weights\n",
    }
    for relative_path, payload in payloads.items():
        path = model_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return model_dir, payloads


def _records(payloads: dict[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        relative_path: {"size_bytes": len(payload), "sha256": _sha256(payload)}
        for relative_path, payload in payloads.items()
    }


def _write_lock(
    tmp_path: Path,
    files: dict[str, dict[str, object]],
    *,
    repository_id: object = REPOSITORY_ID,
    revision: object = REVISION,
    parameter_count: object = PARAMETER_COUNT,
) -> Path:
    lock_path = tmp_path / "formal-model.lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "repository_id": repository_id,
                    "revision": revision,
                    "parameter_count": parameter_count,
                    "files": files,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return lock_path


def _inspect(lock_path: Path, model_dir: Path) -> ModelIdentityFacts:
    return inspect_model_identity(
        lock_path,
        model_dir=model_dir,
        repository_id=REPOSITORY_ID,
        revision=REVISION,
        parameter_count=PARAMETER_COUNT,
    )


def test_load_and_require_model_identity_accept_complete_matching_install(tmp_path: Path) -> None:
    model_dir, payloads = _model_dir(tmp_path)
    lock_path = _write_lock(tmp_path, _records(payloads))

    model_lock = load_model_lock(lock_path)
    facts = _inspect(lock_path, model_dir)

    assert model_lock.repository_id == REPOSITORY_ID
    assert model_lock.revision == REVISION
    assert model_lock.parameter_count == PARAMETER_COUNT
    assert list(model_lock.files) == sorted(payloads)
    assert facts.matches_lock is True
    assert facts.issues == []
    assert facts.extra_safetensors == []
    assert {fact.status for fact in facts.files} == {ModelFileStatus.MATCH}
    assert (
        require_model_identity(
            model_lock,
            model_dir=model_dir,
            repository_id=REPOSITORY_ID,
            revision=REVISION,
            parameter_count=PARAMETER_COUNT,
        )
        == facts
    )


@pytest.mark.parametrize(
    "relative_path,replacement,status,error",
    [
        (
            "tokenizer.json",
            b'{"tokenizer":"no"}\n',
            ModelFileStatus.SHA256_MISMATCH,
            "SHA-256 mismatch",
        ),
        (
            "config.json",
            b'{"model_type":"changed"}\n',
            ModelFileStatus.SIZE_MISMATCH,
            "size mismatch",
        ),
    ],
)
def test_inspect_model_identity_rejects_tampered_locked_file(
    tmp_path: Path,
    relative_path: str,
    replacement: bytes,
    status: ModelFileStatus,
    error: str,
) -> None:
    model_dir, payloads = _model_dir(tmp_path)
    lock_path = _write_lock(tmp_path, _records(payloads))
    (model_dir / relative_path).write_bytes(replacement)

    facts = _inspect(lock_path, model_dir)

    target = next(fact for fact in facts.files if fact.relative_path == relative_path)
    assert facts.matches_lock is False
    assert target.status == status
    assert any(error in issue for issue in facts.issues)


def test_require_model_identity_rejects_missing_locked_file_with_facts(tmp_path: Path) -> None:
    model_dir, payloads = _model_dir(tmp_path)
    lock_path = _write_lock(tmp_path, _records(payloads))
    (model_dir / "tokenizer.json").unlink()

    with pytest.raises(ModelIdentityError, match="locked model file is missing") as error_info:
        require_model_identity(
            lock_path,
            model_dir=model_dir,
            repository_id=REPOSITORY_ID,
            revision=REVISION,
            parameter_count=PARAMETER_COUNT,
        )

    facts = error_info.value.facts
    missing = next(fact for fact in facts.files if fact.relative_path == "tokenizer.json")
    assert facts.matches_lock is False
    assert missing.status == ModelFileStatus.MISSING


def test_inspect_model_identity_rejects_extra_unlocked_safetensors(tmp_path: Path) -> None:
    model_dir, payloads = _model_dir(tmp_path)
    lock_path = _write_lock(tmp_path, _records(payloads))
    extra = model_dir / "nested" / "unlocked.safetensors"
    extra.parent.mkdir()
    extra.write_bytes(b"unlocked weights\n")

    facts = _inspect(lock_path, model_dir)

    assert facts.matches_lock is False
    assert facts.extra_safetensors == ["nested/unlocked.safetensors"]
    assert any("unlocked safetensors" in issue for issue in facts.issues)


def test_require_model_identity_rejects_repository_revision_and_parameter_mismatch(
    tmp_path: Path,
) -> None:
    model_dir, payloads = _model_dir(tmp_path)
    lock_path = _write_lock(tmp_path, _records(payloads))

    with pytest.raises(ModelIdentityError, match="repository mismatch") as error_info:
        require_model_identity(
            lock_path,
            model_dir=model_dir,
            repository_id="Qwen/Different-Model",
            revision="f" * 40,
            parameter_count=PARAMETER_COUNT + 1,
        )

    facts = error_info.value.facts
    assert facts.matches_lock is False
    assert any("repository mismatch" in issue for issue in facts.issues)
    assert any("revision mismatch" in issue for issue in facts.issues)
    assert any("parameter count mismatch" in issue for issue in facts.issues)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../config.json", "/absolute/config.json", "nested\\config.json", "nested/../config.json"],
)
def test_load_model_lock_rejects_unsafe_file_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    _, payloads = _model_dir(tmp_path)
    files = _records(payloads)
    config_record = files.pop("config.json")
    files[unsafe_path] = config_record
    lock_path = _write_lock(tmp_path, files)

    with pytest.raises(ValueError, match="model lock file path"):
        load_model_lock(lock_path)


@pytest.mark.parametrize(
    "repository_id,revision,parameter_count,error",
    [
        (REPOSITORY_ID, "main", PARAMETER_COUNT, "40-character commit"),
        (REPOSITORY_ID, REVISION, True, "positive integer"),
        ("not-a-repository", REVISION, PARAMETER_COUNT, "owner/repository"),
    ],
)
def test_load_model_lock_rejects_invalid_declared_identity(
    tmp_path: Path,
    repository_id: object,
    revision: object,
    parameter_count: object,
    error: str,
) -> None:
    _, payloads = _model_dir(tmp_path)
    lock_path = _write_lock(
        tmp_path,
        _records(payloads),
        repository_id=repository_id,
        revision=revision,
        parameter_count=parameter_count,
    )

    with pytest.raises(ValueError, match=error):
        load_model_lock(lock_path)


@pytest.mark.parametrize("missing_path", ["config.json", "tokenizer.json", "weights"])
def test_load_model_lock_requires_config_tokenizer_and_safetensors(
    tmp_path: Path,
    missing_path: str,
) -> None:
    _, payloads = _model_dir(tmp_path)
    files = _records(payloads)
    if missing_path == "weights":
        files = {
            path: record for path, record in files.items() if not path.endswith(".safetensors")
        }
        error = "at least one safetensors"
    else:
        files.pop(missing_path)
        error = "missing required files"
    lock_path = _write_lock(tmp_path, files)

    with pytest.raises(ValueError, match=error):
        load_model_lock(lock_path)
