"""Unit tests for long-context v5 M0 root artifact integrity sealing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vllm_tuner.longctx.integrity import (
    M0_INTEGRITY_FILE,
    M0_INTEGRITY_SCHEMA,
    seal_m0_artifacts,
    validate_m0_artifacts,
)


def _artifact_root(tmp_path: Path, experiment_id: str = "m0-canary") -> Path:
    root = tmp_path / experiment_id
    (root / "nested").mkdir(parents=True)
    (root / "server.log").write_text("ready\n", encoding="utf-8")
    (root / "nested" / "requests.jsonl").write_text('{"request_id":"one"}\n', encoding="utf-8")
    return root


def _attestation(experiment_id: str = "m0-canary") -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "project_line": "longctx-v5",
        "milestone": "M0",
        "profile": "production-default",
        "source_commit": "a" * 40,
        "runtime_upstream_commit": "b" * 40,
    }


def _read_seal(root: Path) -> dict[str, object]:
    value = json.loads((root / M0_INTEGRITY_FILE).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_seal(root: Path, payload: dict[str, object]) -> None:
    (root / M0_INTEGRITY_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_seal_and_validate_m0_artifacts_cover_nested_regular_files(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)

    sealed = seal_m0_artifacts(
        root,
        experiment_id="m0-canary",
        attestation=_attestation(),
    )

    assert sealed["schema"] == M0_INTEGRITY_SCHEMA
    assert sealed["experiment_id"] == "m0-canary"
    assert sealed["attestation"] == _attestation()
    assert sealed["files"] == [
        {
            "path": "nested/requests.jsonl",
            "size": 21,
            "sha256": "aab85d34156ec7805f69c361b5d7d160429b373297675a1f6b170a292c0af335",
        },
        {
            "path": "server.log",
            "size": 6,
            "sha256": "ed1a545bb85e55816bbf9566b028b2a0bc456b88f49f6f266c0401048824194b",
        },
    ]
    assert validate_m0_artifacts(root) == sealed
    assert not list(root.glob(f".{M0_INTEGRITY_FILE}.*"))


@pytest.mark.parametrize("tamper", ["added", "deleted", "modified"])
def test_validate_m0_artifacts_rejects_artifact_tree_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    root = _artifact_root(tmp_path)
    seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())

    if tamper == "added":
        (root / "added.txt").write_text("unsealed\n", encoding="utf-8")
        error = "file set mismatch"
    elif tamper == "deleted":
        (root / "server.log").unlink()
        error = "file set mismatch"
    else:
        (root / "server.log").write_text("changed\n", encoding="utf-8")
        error = "checksum mismatch"

    with pytest.raises(ValueError, match=error):
        validate_m0_artifacts(root)


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_m0_artifact_integrity_rejects_symlinks(
    tmp_path: Path,
    target_kind: str,
) -> None:
    root = _artifact_root(tmp_path)
    if target_kind == "file":
        os.symlink(root / "server.log", root / "linked-artifact")
    else:
        os.symlink(root / "nested", root / "linked-directory", target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())


def test_validate_m0_artifacts_rejects_symlink_added_after_seal(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())
    os.symlink(root / "server.log", root / "post-seal-link")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        validate_m0_artifacts(root)


def test_seal_m0_artifacts_rejects_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "empty-experiment"
    root.mkdir()

    with pytest.raises(ValueError, match="empty M0 artifact root"):
        seal_m0_artifacts(root, experiment_id=root.name, attestation={"milestone": "M0"})


def test_seal_m0_artifacts_rejects_root_and_attestation_identity_mismatch(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)

    with pytest.raises(ValueError, match="identity mismatch"):
        seal_m0_artifacts(root, experiment_id="another-experiment", attestation=_attestation())
    with pytest.raises(ValueError, match="identity mismatch"):
        seal_m0_artifacts(
            root,
            experiment_id=root.name,
            attestation=_attestation("another-experiment"),
        )
    assert not (root / M0_INTEGRITY_FILE).exists()


@pytest.mark.parametrize("identity_location", ["seal", "attestation"])
def test_validate_m0_artifacts_rejects_tampered_identity(
    tmp_path: Path,
    identity_location: str,
) -> None:
    root = _artifact_root(tmp_path)
    seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())
    payload = _read_seal(root)
    if identity_location == "seal":
        payload["experiment_id"] = "different"
    else:
        attestation = payload["attestation"]
        assert isinstance(attestation, dict)
        attestation["experiment_id"] = "different"
    _write_seal(root, payload)

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_m0_artifacts(root)


def test_validate_m0_artifacts_rejects_tampered_schema_and_digest(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())
    payload = _read_seal(root)
    payload["schema"] = "legacy-schema"
    _write_seal(root, payload)

    with pytest.raises(ValueError, match="schema"):
        validate_m0_artifacts(root)

    payload["schema"] = M0_INTEGRITY_SCHEMA
    files = payload["files"]
    assert isinstance(files, list)
    first_file = files[0]
    assert isinstance(first_file, dict)
    first_file["sha256"] = "0" * 64
    _write_seal(root, payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_m0_artifacts(root)


@pytest.mark.parametrize("attestation", [{}, {"value": float("nan")}, {"value": Path("x")}])
def test_seal_m0_artifacts_rejects_invalid_attestation(
    tmp_path: Path,
    attestation: dict[str, object],
) -> None:
    root = _artifact_root(tmp_path)

    with pytest.raises(ValueError, match="attestation"):
        seal_m0_artifacts(root, experiment_id=root.name, attestation=attestation)
    assert not (root / M0_INTEGRITY_FILE).exists()


def test_seal_m0_artifacts_refuses_existing_seal(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())

    with pytest.raises(ValueError, match="already sealed"):
        seal_m0_artifacts(root, experiment_id=root.name, attestation=_attestation())
