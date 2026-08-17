"""Root-level integrity sealing for long-context v5 M0 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

M0_INTEGRITY_FILE = "m0-integrity.json"
M0_INTEGRITY_SCHEMA = "longctx-v5-m0-integrity-v1"

_SHA256_LENGTH = 64
_READ_CHUNK_BYTES = 1024 * 1024
_TOP_LEVEL_FIELDS = frozenset({"schema", "experiment_id", "attestation", "files"})
_FILE_FIELDS = frozenset({"path", "size", "sha256"})


@dataclass(frozen=True)
class _ExpectedFile:
    """Validated file evidence read from an integrity seal."""

    size: int
    sha256: str


def _reject_nonfinite_json(value: str) -> NoReturn:
    """Reject JavaScript-style non-finite constants accepted by ``json.loads``."""
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _decode_json_object(text: str, description: str) -> dict[str, object]:
    """Decode one strict JSON object with useful integrity context."""
    try:
        value = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{description} is not valid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _artifact_root(root: str | Path) -> Path:
    """Return an existing real directory, never a symlink or broad implicit path."""
    path = Path(root).expanduser()
    try:
        root_status = path.lstat()
    except OSError as error:
        raise ValueError(f"M0 artifact root is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(root_status.st_mode):
        raise ValueError(f"M0 artifact root must not be a symlink: {path}")
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"M0 artifact root must be a directory: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"M0 artifact root cannot be resolved: {path}: {error}") from error


def _validate_experiment_id(value: object, root: Path) -> str:
    """Bind a portable experiment identifier to the experiment directory name."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("M0 artifact experiment_id must be a non-empty stripped string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("M0 artifact experiment_id must be one portable path component")
    if root.name != value:
        raise ValueError(
            f"M0 artifact identity mismatch: root is {root.name!r}, experiment_id is {value!r}"
        )
    return value


def _normalize_attestation(value: object, experiment_id: str) -> dict[str, object]:
    """Copy a non-empty, strictly JSON-serializable attestation and bind its identity."""
    if not isinstance(value, Mapping):
        raise ValueError("M0 artifact attestation must be a non-empty JSON object")
    supplied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("M0 artifact attestation keys must be non-empty strings")
        supplied[key] = item
    if not supplied:
        raise ValueError("M0 artifact attestation must be a non-empty JSON object")
    try:
        encoded = json.dumps(
            supplied,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"M0 artifact attestation is not JSON-serializable: {error}") from error
    normalized = _decode_json_object(encoded, "M0 artifact attestation")
    attested_experiment_id = normalized.get("experiment_id")
    if attested_experiment_id is not None and attested_experiment_id != experiment_id:
        raise ValueError(
            "M0 artifact identity mismatch: attestation experiment_id does not match the seal"
        )
    return normalized


def _regular_artifact_files(root: Path) -> dict[str, Path]:
    """Enumerate the exact regular-file tree without following any symlink."""
    files: dict[str, Path] = {}
    directories = [root]
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError as error:
            raise ValueError(
                f"unable to enumerate M0 artifact directory {directory}: {error}"
            ) from error
        for entry in entries:
            path = directory / entry.name
            try:
                entry_status = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"unable to inspect M0 artifact path {path}: {error}") from error
            if stat.S_ISLNK(entry_status.st_mode):
                raise ValueError(f"M0 artifact tree must not contain symlinks: {path}")
            if stat.S_ISDIR(entry_status.st_mode):
                directories.append(path)
                continue
            if not stat.S_ISREG(entry_status.st_mode):
                raise ValueError(f"M0 artifact tree contains a non-regular file: {path}")
            relative_path = path.relative_to(root).as_posix()
            if relative_path == M0_INTEGRITY_FILE:
                continue
            files[relative_path] = path
    return files


def _hash_regular_file(path: Path) -> tuple[int, str]:
    """Hash a stable regular file through a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"unable to open M0 artifact file {path}: {error}") from error
    digest = hashlib.sha256()
    total_size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"M0 artifact path is no longer a regular file: {path}")
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total_size += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"unable to hash M0 artifact file {path}: {error}") from error
    finally:
        os.close(descriptor)

    stable_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    hashed_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    try:
        current = path.lstat()
    except OSError as error:
        raise ValueError(f"M0 artifact changed while hashing: {path}: {error}") from error
    current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if (
        stable_identity != hashed_identity
        or hashed_identity != current_identity
        or total_size != after.st_size
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError(f"M0 artifact changed while hashing: {path}")
    return total_size, digest.hexdigest()


def _file_records(files: Mapping[str, Path]) -> list[dict[str, object]]:
    """Build deterministic size and SHA-256 records for an enumerated artifact tree."""
    records: list[dict[str, object]] = []
    for relative_path, path in sorted(files.items()):
        size, sha256 = _hash_regular_file(path)
        records.append({"path": relative_path, "size": size, "sha256": sha256})
    return records


def _validate_relative_path(value: object) -> str:
    """Validate one canonical relative POSIX artifact path."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("M0 integrity file paths must be non-empty relative POSIX paths")
    pure_path = PurePosixPath(value)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or value == M0_INTEGRITY_FILE
    ):
        raise ValueError(f"M0 integrity file path is not canonical and relative: {value!r}")
    return value


def _expected_files(value: object) -> tuple[list[dict[str, object]], dict[str, _ExpectedFile]]:
    """Validate sealed file records and return their comparison index."""
    if not isinstance(value, list) or not value:
        raise ValueError("M0 integrity file list must be non-empty")
    records: list[dict[str, object]] = []
    expected: dict[str, _ExpectedFile] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != _FILE_FIELDS:
            raise ValueError("M0 integrity file records require exactly path, size, and sha256")
        relative_path = _validate_relative_path(item.get("path"))
        size = item.get("size")
        sha256 = item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"M0 integrity file size is invalid: {relative_path}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"M0 integrity SHA-256 is invalid: {relative_path}")
        if relative_path in expected:
            raise ValueError(f"M0 integrity file path is duplicated: {relative_path}")
        record = {"path": relative_path, "size": size, "sha256": sha256}
        records.append(record)
        expected[relative_path] = _ExpectedFile(size=size, sha256=sha256)
    if [record["path"] for record in records] != sorted(expected):
        raise ValueError("M0 integrity file records must be sorted by relative POSIX path")
    return records, expected


def _validated_payload(
    payload: dict[str, object],
    root: Path,
) -> tuple[dict[str, object], dict[str, _ExpectedFile]]:
    """Validate seal metadata before trusting any checksum evidence."""
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError(
            "M0 integrity metadata requires exactly schema, experiment_id, attestation, files"
        )
    if payload.get("schema") != M0_INTEGRITY_SCHEMA:
        raise ValueError("M0 integrity schema does not match the v5 M0 contract")
    experiment_id = _validate_experiment_id(payload.get("experiment_id"), root)
    attestation = _normalize_attestation(payload.get("attestation"), experiment_id)
    records, expected = _expected_files(payload.get("files"))
    normalized: dict[str, object] = {
        "schema": M0_INTEGRITY_SCHEMA,
        "experiment_id": experiment_id,
        "attestation": attestation,
        "files": records,
    }
    return normalized, expected


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON beside its final path and atomically replace the destination."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def seal_m0_artifacts(
    root: str | Path,
    *,
    experiment_id: str,
    attestation: Mapping[str, object],
) -> dict[str, object]:
    """Atomically seal every regular artifact below one v5 M0 experiment root."""
    artifact_root = _artifact_root(root)
    validated_experiment_id = _validate_experiment_id(experiment_id, artifact_root)
    normalized_attestation = _normalize_attestation(attestation, validated_experiment_id)
    integrity_path = artifact_root / M0_INTEGRITY_FILE
    if integrity_path.is_symlink():
        raise ValueError(f"M0 artifact tree must not contain symlinks: {integrity_path}")
    if integrity_path.exists():
        raise ValueError(f"M0 artifact root is already sealed: {integrity_path}")

    files = _regular_artifact_files(artifact_root)
    if not files:
        raise ValueError("cannot seal an empty M0 artifact root")
    payload: dict[str, object] = {
        "schema": M0_INTEGRITY_SCHEMA,
        "experiment_id": validated_experiment_id,
        "attestation": normalized_attestation,
        "files": _file_records(files),
    }
    _atomic_json(integrity_path, payload)
    return payload


def validate_m0_artifacts(root: str | Path) -> dict[str, object]:
    """Reject an invalid seal or any added, deleted, modified, or linked artifact."""
    artifact_root = _artifact_root(root)
    integrity_path = artifact_root / M0_INTEGRITY_FILE
    if integrity_path.is_symlink() or not integrity_path.is_file():
        raise ValueError(f"M0 artifact root has no regular {M0_INTEGRITY_FILE}: {artifact_root}")
    try:
        payload = _decode_json_object(
            integrity_path.read_text(encoding="utf-8"),
            M0_INTEGRITY_FILE,
        )
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to read M0 integrity seal {integrity_path}: {error}") from error
    normalized, expected = _validated_payload(payload, artifact_root)

    actual_files = _regular_artifact_files(artifact_root)
    actual_names = set(actual_files)
    expected_names = set(expected)
    if actual_names != expected_names:
        added = sorted(actual_names - expected_names)
        deleted = sorted(expected_names - actual_names)
        details: list[str] = []
        if added:
            details.append("added=" + ",".join(added))
        if deleted:
            details.append("deleted=" + ",".join(deleted))
        raise ValueError("M0 artifact file set mismatch: " + "; ".join(details))

    for relative_path, path in sorted(actual_files.items()):
        size, sha256 = _hash_regular_file(path)
        evidence = expected[relative_path]
        if size != evidence.size or sha256 != evidence.sha256:
            raise ValueError(f"M0 artifact checksum mismatch: {relative_path}")
    return normalized
