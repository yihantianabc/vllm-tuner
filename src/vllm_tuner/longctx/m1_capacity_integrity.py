"""Independent root integrity sealing for long-context v5 M1 capacity artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

M1_CAPACITY_INTEGRITY_FILE = "m1-capacity-integrity.json"
M1_CAPACITY_INTEGRITY_SCHEMA = "longctx-v5-m1-capacity-integrity-v1"

_SHA256_LENGTH = 64
_READ_CHUNK_BYTES = 1024 * 1024
_PORTABLE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_LEVEL_FIELDS = frozenset({"schema", "experiment_id", "attestation", "files"})
_FILE_FIELDS = frozenset({"path", "size", "sha256"})


@dataclass(frozen=True)
class _ExpectedFile:
    """Validated file evidence read from an M1 capacity seal."""

    size: int
    sha256: str


def _reject_nonfinite_json(value: str) -> NoReturn:
    """Reject JavaScript-style non-finite constants accepted by ``json.loads``."""
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    """Reject numeric JSON literals whose conversion overflows to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _decode_json_object(text: str, description: str) -> dict[str, object]:
    """Decode exactly one strict, finite JSON object."""
    try:
        value = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
                parse_float=_parse_finite_float,
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{description} is not valid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _artifact_root(root: str | Path) -> Path:
    """Return an existing real directory, never a symlink or implicit broad path."""
    path = Path(root).expanduser()
    try:
        root_status = path.lstat()
    except OSError as error:
        raise ValueError(f"M1 capacity artifact root is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(root_status.st_mode):
        raise ValueError(f"M1 capacity artifact root must not be a symlink: {path}")
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"M1 capacity artifact root must be a directory: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"M1 capacity artifact root cannot be resolved: {path}: {error}"
        ) from error


def _validate_experiment_id(value: object, root: Path) -> str:
    """Bind a portable experiment identifier to the experiment directory name."""
    if not isinstance(value, str) or _PORTABLE_EXPERIMENT_ID.fullmatch(value) is None:
        raise ValueError("M1 capacity experiment_id must be one portable path component")
    if root.name != value:
        raise ValueError(
            "M1 capacity artifact identity mismatch: "
            f"root is {root.name!r}, experiment_id is {value!r}"
        )
    return value


def _normalize_attestation(value: object, experiment_id: str) -> dict[str, object]:
    """Copy a non-empty strict JSON attestation and bind any declared identity."""
    if not isinstance(value, Mapping):
        raise ValueError("M1 capacity attestation must be a non-empty JSON object")
    supplied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("M1 capacity attestation keys must be non-empty strings")
        supplied[key] = item
    if not supplied:
        raise ValueError("M1 capacity attestation must be a non-empty JSON object")
    try:
        encoded = json.dumps(
            supplied,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"M1 capacity attestation is not strict JSON: {error}") from error
    normalized = _decode_json_object(encoded, "M1 capacity attestation")
    attested_experiment_id = normalized.get("experiment_id")
    if attested_experiment_id is not None and attested_experiment_id != experiment_id:
        raise ValueError(
            "M1 capacity artifact identity mismatch: "
            "attestation experiment_id does not match the seal"
        )
    return normalized


def _regular_artifact_files(root: Path) -> dict[str, Path]:
    """Enumerate the exact regular-file tree without following a symlink."""
    files: dict[str, Path] = {}
    directories = [root]
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError as error:
            raise ValueError(
                f"unable to enumerate M1 capacity artifact directory {directory}: {error}"
            ) from error
        for entry in entries:
            path = directory / entry.name
            try:
                entry_status = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"unable to inspect M1 capacity artifact path {path}: {error}"
                ) from error
            if stat.S_ISLNK(entry_status.st_mode):
                raise ValueError(f"M1 capacity artifact tree must not contain symlinks: {path}")
            if stat.S_ISDIR(entry_status.st_mode):
                directories.append(path)
                continue
            if not stat.S_ISREG(entry_status.st_mode):
                raise ValueError(f"M1 capacity artifact tree contains a non-regular file: {path}")
            relative_path = path.relative_to(root).as_posix()
            if relative_path == M1_CAPACITY_INTEGRITY_FILE:
                continue
            files[relative_path] = path
    return files


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return fields that expose replacement or mutation across one read."""
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _open_regular_no_follow(path: Path, description: str) -> tuple[int, os.stat_result]:
    """Open a regular file without following its final path component."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"unable to open {description} {path}: {error}") from error
    try:
        opened_status = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"unable to inspect {description} {path}: {error}") from error
    if not stat.S_ISREG(opened_status.st_mode):
        os.close(descriptor)
        raise ValueError(f"{description} is not a regular file: {path}")
    return descriptor, opened_status


def _stable_current_status(
    path: Path,
    before: os.stat_result,
    after: os.stat_result,
    description: str,
) -> None:
    """Require a descriptor and its path to retain one regular-file identity."""
    try:
        current = path.lstat()
    except OSError as error:
        raise ValueError(f"{description} changed while reading: {path}: {error}") from error
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(current)
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError(f"{description} changed while reading: {path}")


def _hash_regular_file(path: Path) -> tuple[int, str]:
    """Hash a stable regular artifact through a no-follow descriptor."""
    descriptor, before = _open_regular_no_follow(path, "M1 capacity artifact file")
    digest = hashlib.sha256()
    total_size = 0
    try:
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total_size += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"unable to hash M1 capacity artifact file {path}: {error}") from error
    finally:
        os.close(descriptor)
    _stable_current_status(path, before, after, "M1 capacity artifact file")
    if total_size != after.st_size:
        raise ValueError(f"M1 capacity artifact file changed while reading: {path}")
    return total_size, digest.hexdigest()


def _read_regular_text(path: Path) -> str:
    """Read the integrity seal through the same stable no-follow discipline."""
    descriptor, before = _open_regular_no_follow(path, "M1 capacity integrity seal")
    chunks: list[bytes] = []
    total_size = 0
    try:
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            total_size += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"unable to read M1 capacity integrity seal {path}: {error}") from error
    finally:
        os.close(descriptor)
    _stable_current_status(path, before, after, "M1 capacity integrity seal")
    if total_size != after.st_size:
        raise ValueError(f"M1 capacity integrity seal changed while reading: {path}")
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"M1 capacity integrity seal is not UTF-8: {path}: {error}") from error


def _file_records(files: Mapping[str, Path]) -> list[dict[str, object]]:
    """Build deterministic size and SHA-256 records for an artifact tree."""
    records: list[dict[str, object]] = []
    for relative_path, path in sorted(files.items()):
        size, sha256 = _hash_regular_file(path)
        records.append({"path": relative_path, "size": size, "sha256": sha256})
    return records


def _validate_relative_path(value: object) -> str:
    """Validate one canonical relative POSIX artifact path."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("M1 capacity integrity paths must be non-empty relative POSIX paths")
    pure_path = PurePosixPath(value)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or value == M1_CAPACITY_INTEGRITY_FILE
    ):
        raise ValueError(f"M1 capacity integrity path is not canonical and relative: {value!r}")
    return value


def _expected_files(value: object) -> tuple[list[dict[str, object]], dict[str, _ExpectedFile]]:
    """Validate sealed file records and return their comparison index."""
    if not isinstance(value, list) or not value:
        raise ValueError("M1 capacity integrity file list must be non-empty")
    records: list[dict[str, object]] = []
    expected: dict[str, _ExpectedFile] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != _FILE_FIELDS:
            raise ValueError("M1 capacity integrity records require exactly path, size, and sha256")
        relative_path = _validate_relative_path(item.get("path"))
        size = item.get("size")
        sha256 = item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"M1 capacity integrity size is invalid: {relative_path}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"M1 capacity integrity SHA-256 is invalid: {relative_path}")
        if relative_path in expected:
            raise ValueError(f"M1 capacity integrity path is duplicated: {relative_path}")
        record = {"path": relative_path, "size": size, "sha256": sha256}
        records.append(record)
        expected[relative_path] = _ExpectedFile(size=size, sha256=sha256)
    if [record["path"] for record in records] != sorted(expected):
        raise ValueError("M1 capacity integrity records must be sorted by relative POSIX path")
    return records, expected


def _validated_payload(
    payload: dict[str, object],
    root: Path,
) -> tuple[dict[str, object], dict[str, _ExpectedFile]]:
    """Validate seal metadata before trusting checksum evidence."""
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError(
            "M1 capacity integrity metadata requires exactly "
            "schema, experiment_id, attestation, files"
        )
    if payload.get("schema") != M1_CAPACITY_INTEGRITY_SCHEMA:
        raise ValueError("M1 capacity integrity schema does not match the v5 contract")
    experiment_id = _validate_experiment_id(payload.get("experiment_id"), root)
    attestation = _normalize_attestation(payload.get("attestation"), experiment_id)
    records, expected = _expected_files(payload.get("files"))
    normalized: dict[str, object] = {
        "schema": M1_CAPACITY_INTEGRITY_SCHEMA,
        "experiment_id": experiment_id,
        "attestation": attestation,
        "files": records,
    }
    return normalized, expected


def _atomic_json_once(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish a complete JSON seal without replacing any existing entry."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError(f"M1 capacity artifact root is already sealed: {path}") from error
        except OSError as error:
            raise ValueError(
                f"unable to publish M1 capacity integrity seal {path}: {error}"
            ) from error
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def seal_m1_capacity_artifacts(
    root: str | Path,
    experiment_id: str,
    attestation: Mapping[str, object],
) -> dict[str, object]:
    """Atomically seal every regular artifact below one v5 M1 capacity root."""
    artifact_root = _artifact_root(root)
    validated_experiment_id = _validate_experiment_id(experiment_id, artifact_root)
    normalized_attestation = _normalize_attestation(attestation, validated_experiment_id)
    integrity_path = artifact_root / M1_CAPACITY_INTEGRITY_FILE
    if integrity_path.is_symlink() or integrity_path.exists():
        raise ValueError(f"M1 capacity artifact root is already sealed: {integrity_path}")

    files = _regular_artifact_files(artifact_root)
    if not files:
        raise ValueError("cannot seal an empty M1 capacity artifact root")
    payload: dict[str, object] = {
        "schema": M1_CAPACITY_INTEGRITY_SCHEMA,
        "experiment_id": validated_experiment_id,
        "attestation": normalized_attestation,
        "files": _file_records(files),
    }
    _atomic_json_once(integrity_path, payload)
    return payload


def validate_m1_capacity_artifacts(root: str | Path) -> dict[str, object]:
    """Reject an invalid seal or any added, deleted, modified, or linked artifact."""
    artifact_root = _artifact_root(root)
    integrity_path = artifact_root / M1_CAPACITY_INTEGRITY_FILE
    payload = _decode_json_object(
        _read_regular_text(integrity_path),
        M1_CAPACITY_INTEGRITY_FILE,
    )
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
        raise ValueError("M1 capacity artifact file set mismatch: " + "; ".join(details))

    for relative_path, path in sorted(actual_files.items()):
        size, sha256 = _hash_regular_file(path)
        evidence = expected[relative_path]
        if size != evidence.size or sha256 != evidence.sha256:
            raise ValueError(f"M1 capacity artifact checksum mismatch: {relative_path}")
    return normalized
