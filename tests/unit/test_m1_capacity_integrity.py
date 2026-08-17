"""Tests for the independent long-context v5 M1 capacity root seal."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vllm_tuner.longctx.m1_capacity_integrity import (
    M1_CAPACITY_INTEGRITY_FILE,
    M1_CAPACITY_INTEGRITY_SCHEMA,
    seal_m1_capacity_artifacts,
    validate_m1_capacity_artifacts,
)


def _artifact_root(tmp_path: Path, experiment_id: str = "m1-capacity-test") -> Path:
    root = tmp_path / experiment_id
    (root / "trials" / "context-8k-low-0").mkdir(parents=True)
    (root / "manifest.json").write_text('{"project_line":"longctx-v5"}\n', encoding="utf-8")
    (root / "trials" / "context-8k-low-0" / "requests.jsonl").write_text(
        '{"request_id":"one"}\n',
        encoding="utf-8",
    )
    return root


def _attestation(experiment_id: str = "m1-capacity-test") -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "project_line": "longctx-v5",
        "milestone": "M1",
        "evidence_role": "formal",
        "source_commit": "a" * 40,
        "initialization_artifact_sha256": "b" * 64,
        "acceptance_passed": True,
    }


def _read_seal(root: Path) -> dict[str, object]:
    value = json.loads((root / M1_CAPACITY_INTEGRITY_FILE).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_seal(root: Path, payload: dict[str, object]) -> None:
    (root / M1_CAPACITY_INTEGRITY_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_seal_and_validate_cover_exact_nested_regular_file_tree(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)

    sealed = seal_m1_capacity_artifacts(root, root.name, _attestation())

    assert sealed["schema"] == M1_CAPACITY_INTEGRITY_SCHEMA
    assert sealed["experiment_id"] == root.name
    assert sealed["attestation"] == _attestation()
    assert [record["path"] for record in sealed["files"]] == [
        "manifest.json",
        "trials/context-8k-low-0/requests.jsonl",
    ]
    assert validate_m1_capacity_artifacts(root) == sealed
    assert sorted(path.name for path in root.iterdir()) == [
        M1_CAPACITY_INTEGRITY_FILE,
        "manifest.json",
        "trials",
    ]
    assert not list(root.glob(f".{M1_CAPACITY_INTEGRITY_FILE}.*"))


@pytest.mark.parametrize("tamper", ["added", "deleted", "modified"])
def test_validate_rejects_changes_after_sealing(tmp_path: Path, tamper: str) -> None:
    root = _artifact_root(tmp_path)
    seal_m1_capacity_artifacts(root, root.name, _attestation())

    if tamper == "added":
        (root / "late-status.json").write_text("{}\n", encoding="utf-8")
        error = "file set mismatch"
    elif tamper == "deleted":
        (root / "manifest.json").unlink()
        error = "file set mismatch"
    else:
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
        error = "checksum mismatch"

    with pytest.raises(ValueError, match=error):
        validate_m1_capacity_artifacts(root)


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_seal_rejects_symlinks_in_artifact_tree(tmp_path: Path, target_kind: str) -> None:
    root = _artifact_root(tmp_path)
    if target_kind == "file":
        os.symlink(root / "manifest.json", root / "linked-artifact")
    else:
        os.symlink(root / "trials", root / "linked-directory", target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        seal_m1_capacity_artifacts(root, root.name, _attestation())


def test_validate_rejects_symlink_added_after_seal(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    seal_m1_capacity_artifacts(root, root.name, _attestation())
    os.symlink(root / "manifest.json", root / "late-link")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        validate_m1_capacity_artifacts(root)


def test_root_and_integrity_entry_must_be_real_regular_paths(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    linked_root = tmp_path / "linked-root"
    os.symlink(root, linked_root, target_is_directory=True)
    with pytest.raises(ValueError, match="root must not be a symlink"):
        seal_m1_capacity_artifacts(linked_root, root.name, _attestation())

    os.symlink(root / "missing-seal", root / M1_CAPACITY_INTEGRITY_FILE)
    with pytest.raises(ValueError, match="already sealed"):
        seal_m1_capacity_artifacts(root, root.name, _attestation())
    with pytest.raises(ValueError, match="unable to open M1 capacity integrity seal"):
        validate_m1_capacity_artifacts(root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_seal_rejects_non_regular_artifact(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    os.mkfifo(root / "telemetry.pipe")

    with pytest.raises(ValueError, match="non-regular file"):
        seal_m1_capacity_artifacts(root, root.name, _attestation())


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_validate_rejects_non_regular_integrity_entry_without_blocking(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    os.mkfifo(root / M1_CAPACITY_INTEGRITY_FILE)

    with pytest.raises(ValueError, match="integrity seal is not a regular file"):
        validate_m1_capacity_artifacts(root)


def test_seal_rejects_empty_root_and_cannot_reseal(tmp_path: Path) -> None:
    empty = tmp_path / "m1-capacity-empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty M1 capacity artifact root"):
        seal_m1_capacity_artifacts(empty, empty.name, {"milestone": "M1"})

    root = _artifact_root(tmp_path)
    sealed = seal_m1_capacity_artifacts(root, root.name, _attestation())
    with pytest.raises(ValueError, match="already sealed"):
        seal_m1_capacity_artifacts(root, root.name, _attestation())
    assert validate_m1_capacity_artifacts(root) == sealed


def test_seal_rejects_root_and_attestation_identity_mismatch(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)

    with pytest.raises(ValueError, match="identity mismatch"):
        seal_m1_capacity_artifacts(root, "another-experiment", _attestation())
    with pytest.raises(ValueError, match="identity mismatch"):
        seal_m1_capacity_artifacts(root, root.name, _attestation("another-experiment"))
    assert not (root / M1_CAPACITY_INTEGRITY_FILE).exists()


@pytest.mark.parametrize("identity_location", ["seal", "attestation"])
def test_validate_rejects_tampered_identity(tmp_path: Path, identity_location: str) -> None:
    root = _artifact_root(tmp_path)
    seal_m1_capacity_artifacts(root, root.name, _attestation())
    payload = _read_seal(root)
    if identity_location == "seal":
        payload["experiment_id"] = "different"
    else:
        attestation = payload["attestation"]
        assert isinstance(attestation, dict)
        attestation["experiment_id"] = "different"
    _write_seal(root, payload)

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_m1_capacity_artifacts(root)


def test_validate_rejects_schema_digest_and_file_record_tampering(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    seal_m1_capacity_artifacts(root, root.name, _attestation())
    payload = _read_seal(root)
    payload["schema"] = "legacy-schema"
    _write_seal(root, payload)
    with pytest.raises(ValueError, match="schema"):
        validate_m1_capacity_artifacts(root)

    payload["schema"] = M1_CAPACITY_INTEGRITY_SCHEMA
    files = payload["files"]
    assert isinstance(files, list)
    first_file = files[0]
    assert isinstance(first_file, dict)
    first_file["sha256"] = "0" * 64
    _write_seal(root, payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_m1_capacity_artifacts(root)

    first_file["extra"] = False
    _write_seal(root, payload)
    with pytest.raises(ValueError, match="require exactly"):
        validate_m1_capacity_artifacts(root)


@pytest.mark.parametrize(
    "attestation",
    [{}, {"value": float("nan")}, {"value": Path("x")}],
)
def test_seal_rejects_invalid_attestation(
    tmp_path: Path,
    attestation: dict[str, object],
) -> None:
    root = _artifact_root(tmp_path)

    with pytest.raises(ValueError, match="attestation"):
        seal_m1_capacity_artifacts(root, root.name, attestation)
    assert not (root / M1_CAPACITY_INTEGRITY_FILE).exists()


@pytest.mark.parametrize(
    "invalid_json",
    [
        '{"schema":"first","schema":"second"}',
        '{"schema":NaN}',
        '{"schema":1e999}',
        "[]",
    ],
)
def test_validate_rejects_non_strict_json(tmp_path: Path, invalid_json: str) -> None:
    root = _artifact_root(tmp_path)
    (root / M1_CAPACITY_INTEGRITY_FILE).write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON|must be a JSON object"):
        validate_m1_capacity_artifacts(root)
