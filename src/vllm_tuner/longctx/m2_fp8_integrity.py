"""Root-level integrity sealing for long-context v5 M2 FP8 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

M2_FP8_INTEGRITY_FILE = "m2-fp8-integrity.json"
M2_FP8_INTEGRITY_SCHEMA = "longctx-v5-m2-fp8-integrity-v1"

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"M2 FP8 artifact root must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"M2 FP8 artifact root must be a directory: {resolved}")
    return resolved


def _artifact_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"M2 FP8 artifacts must not contain symlinks: {path}")
        if not path.is_file() or path.name == M2_FP8_INTEGRITY_FILE:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = path
    return files


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("M2 FP8 integrity path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"invalid M2 FP8 integrity path: {value!r}")
    return value


def _atomic_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"M2 FP8 integrity seal already exists: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seal_m2_fp8_artifacts(
    root: str | Path,
    experiment_id: str,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every current artifact and create one immutable root seal."""
    resolved = _root(root)
    if _PORTABLE_ID.fullmatch(experiment_id) is None or resolved.name != experiment_id:
        raise ValueError("M2 FP8 experiment identity must match its artifact directory")
    files = _artifact_files(resolved)
    required = {"experiment.json", "manifest.json", "status.json", "summary.json"}
    missing = sorted(required - set(files))
    if missing:
        raise ValueError("M2 FP8 artifact root is incomplete: " + ", ".join(missing))
    records = [
        {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(files.items())
    ]
    payload = {
        "schema": M2_FP8_INTEGRITY_SCHEMA,
        "experiment_id": experiment_id,
        "attestation": dict(attestation),
        "files": records,
    }
    _atomic_json_once(resolved / M2_FP8_INTEGRITY_FILE, payload)
    return validate_m2_fp8_artifacts(resolved)


def validate_m2_fp8_artifacts(root: str | Path) -> dict[str, Any]:
    """Verify the exact file set, sizes, hashes, schema, and experiment identity."""
    resolved = _root(root)
    seal = resolved / M2_FP8_INTEGRITY_FILE
    if not seal.is_file() or seal.is_symlink():
        raise ValueError("M2 FP8 artifact root has no integrity seal")
    try:
        payload = json.loads(seal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid M2 FP8 integrity seal: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("M2 FP8 integrity seal must be a JSON object")
    if payload.get("schema") != M2_FP8_INTEGRITY_SCHEMA:
        raise ValueError("M2 FP8 integrity schema mismatch")
    experiment_id = payload.get("experiment_id")
    if (
        not isinstance(experiment_id, str)
        or _PORTABLE_ID.fullmatch(experiment_id) is None
        or experiment_id != resolved.name
    ):
        raise ValueError("M2 FP8 integrity experiment identity mismatch")
    raw_records = payload.get("files")
    if not isinstance(raw_records, list):
        raise ValueError("M2 FP8 integrity files must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            raise ValueError("invalid M2 FP8 integrity file record")
        relative = _validate_relative_path(raw.get("path"))
        size = raw.get("size")
        digest = raw.get("sha256")
        if (
            relative in expected
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"invalid M2 FP8 integrity identity for {relative}")
        expected[relative] = (size, digest)
    actual = _artifact_files(resolved)
    if set(actual) != set(expected):
        raise ValueError("M2 FP8 integrity file set mismatch")
    for relative, path in actual.items():
        size, digest = expected[relative]
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError(f"M2 FP8 artifact checksum mismatch: {relative}")
    return payload


__all__ = [
    "M2_FP8_INTEGRITY_FILE",
    "seal_m2_fp8_artifacts",
    "validate_m2_fp8_artifacts",
]
