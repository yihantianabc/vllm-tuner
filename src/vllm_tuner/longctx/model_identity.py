"""Fail-closed identity verification for locally installed formal models."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_LENGTH = 64
_READ_CHUNK_BYTES = 1024 * 1024
_MODEL_FIELDS = frozenset({"repository_id", "revision", "parameter_count", "files"})
_FILE_FIELDS = frozenset({"size_bytes", "sha256"})
_REQUIRED_MODEL_FILES = frozenset({"config.json", "tokenizer.json"})


class ModelFileStatus(str, Enum):
    """Comparison result for one file named by a formal model lock."""

    MATCH = "match"
    MISSING = "missing"
    NOT_REGULAR = "not_regular"
    SIZE_MISMATCH = "size_mismatch"
    SHA256_MISMATCH = "sha256_mismatch"
    UNAVAILABLE = "unavailable"


class LockedModelFile(BaseModel):
    """Expected byte identity for one file below the model directory."""

    size_bytes: int
    sha256: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentityLock(BaseModel):
    """Pinned repository identity and complete locked-file evidence."""

    lock_path: Path
    repository_id: str
    revision: str
    parameter_count: int
    files: dict[str, LockedModelFile]

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelFileFact(BaseModel):
    """Expected and observed identity for one locked model file."""

    relative_path: str
    absolute_path: Path
    expected_size_bytes: int
    actual_size_bytes: Optional[int] = None
    expected_sha256: str
    actual_sha256: Optional[str] = None
    status: ModelFileStatus
    issue: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentityFacts(BaseModel):
    """Structured evidence for a model lock comparison."""

    lock_path: Path
    model_dir: Path
    expected_repository_id: str
    actual_repository_id: str
    expected_revision: str
    actual_revision: str
    expected_parameter_count: int
    actual_parameter_count: int
    files: list[ModelFileFact] = Field(default_factory=list)
    extra_safetensors: list[str] = Field(default_factory=list)
    matches_lock: bool
    issues: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentityError(ValueError):
    """Raised when local model evidence does not match its formal lock."""

    def __init__(self, facts: ModelIdentityFacts) -> None:
        self.facts = facts
        detail = "; ".join(facts.issues) if facts.issues else "unknown identity mismatch"
        super().__init__(f"model identity check failed: {detail}")


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping without coercing ambiguous YAML keys."""
    if not isinstance(value, Mapping):
        raise ValueError(f"model lock field '{field}' must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"model lock field '{field}' requires non-empty string keys")
        result[key] = item
    return result


def _exact_fields(mapping: Mapping[str, object], expected: frozenset[str], field: str) -> None:
    """Reject omitted and unknown fields instead of silently weakening a lock."""
    actual = set(mapping)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if extra:
        details.append("extra=" + ",".join(extra))
    raise ValueError(f"model lock field '{field}' has invalid fields: {'; '.join(details)}")


def _nonempty_string(value: object, field: str) -> str:
    """Return one already-stripped non-empty string."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"model lock field '{field}' must be a non-empty stripped string")
    return value


def _positive_integer(value: object, field: str) -> int:
    """Reject booleans, numeric coercions, and non-positive integer identities."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"model lock field '{field}' must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    """Reject booleans, numeric coercions, and negative byte sizes."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"model lock field '{field}' must be a non-negative integer")
    return value


def _sha256(value: object, field: str) -> str:
    """Validate and normalize one SHA-256 digest."""
    digest = _nonempty_string(value, field).lower()
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"model lock field '{field}' must be a 64-character SHA-256")
    return digest


def _relative_model_path(value: object) -> str:
    """Return one canonical relative POSIX path that cannot escape the model root."""
    path = _nonempty_string(value, "model.files path")
    if "\\" in path or "\x00" in path:
        raise ValueError(f"model lock file path must use safe POSIX separators: {path!r}")
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError(f"model lock file path is not canonical and relative: {path!r}")
    return path


def _locked_file(value: object, relative_path: str) -> LockedModelFile:
    """Parse one exact size-and-digest record."""
    record = _mapping(value, f"model.files.{relative_path}")
    _exact_fields(record, _FILE_FIELDS, f"model.files.{relative_path}")
    return LockedModelFile(
        size_bytes=_nonnegative_integer(
            record.get("size_bytes"), f"model.files.{relative_path}.size_bytes"
        ),
        sha256=_sha256(record.get("sha256"), f"model.files.{relative_path}.sha256"),
    )


def load_model_lock(lock_path: str | Path) -> ModelIdentityLock:
    """Load a strict formal-model YAML lock with complete byte evidence."""
    resolved_lock_path = Path(lock_path).expanduser().resolve(strict=False)
    if resolved_lock_path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("formal model lock must use a .yaml or .yml suffix")
    try:
        payload = yaml.safe_load(resolved_lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(
            f"unable to read formal model lock {resolved_lock_path}: {error}"
        ) from error

    root = _mapping(payload, "root")
    _exact_fields(root, frozenset({"model"}), "root")
    model = _mapping(root.get("model"), "model")
    _exact_fields(model, _MODEL_FIELDS, "model")

    repository_id = _nonempty_string(model.get("repository_id"), "model.repository_id")
    if _REPOSITORY_PATTERN.fullmatch(repository_id) is None:
        raise ValueError("model lock field 'model.repository_id' must use owner/repository form")
    revision = _nonempty_string(model.get("revision"), "model.revision")
    if _COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError("model lock field 'model.revision' must be a 40-character commit")
    parameter_count = _positive_integer(model.get("parameter_count"), "model.parameter_count")

    raw_files = _mapping(model.get("files"), "model.files")
    files: dict[str, LockedModelFile] = {}
    for raw_path, raw_record in raw_files.items():
        relative_path = _relative_model_path(raw_path)
        if relative_path in files:
            raise ValueError(f"model lock file path is duplicated: {relative_path}")
        files[relative_path] = _locked_file(raw_record, relative_path)
    missing_required = sorted(_REQUIRED_MODEL_FILES - set(files))
    if missing_required:
        raise ValueError("model lock is missing required files: " + ", ".join(missing_required))
    if not any(PurePosixPath(path).suffix.casefold() == ".safetensors" for path in files):
        raise ValueError("model lock must contain at least one safetensors weight file")

    return ModelIdentityLock(
        lock_path=resolved_lock_path,
        repository_id=repository_id,
        revision=revision,
        parameter_count=parameter_count,
        files=dict(sorted(files.items())),
    )


def _file_fact(
    relative_path: str,
    absolute_path: Path,
    evidence: LockedModelFile,
    *,
    status: ModelFileStatus,
    actual_size_bytes: Optional[int] = None,
    actual_sha256: Optional[str] = None,
    issue: Optional[str] = None,
) -> ModelFileFact:
    """Construct one consistently populated file fact."""
    return ModelFileFact(
        relative_path=relative_path,
        absolute_path=absolute_path,
        expected_size_bytes=evidence.size_bytes,
        actual_size_bytes=actual_size_bytes,
        expected_sha256=evidence.sha256,
        actual_sha256=actual_sha256,
        status=status,
        issue=issue,
    )


def _hash_regular_file(path: Path) -> tuple[int, str]:
    """Hash a stable regular file through a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    total_size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"path is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    current = path.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if (
        before_identity != after_identity
        or after_identity != current_identity
        or total_size != after.st_size
        or not stat.S_ISREG(current.st_mode)
    ):
        raise OSError(f"file changed while hashing: {path}")
    return total_size, digest.hexdigest()


def _inspect_locked_file(
    model_dir: Path,
    relative_path: str,
    evidence: LockedModelFile,
) -> ModelFileFact:
    """Compare one locked path without following symlink components."""
    pure_path = PurePosixPath(relative_path)
    absolute_path = model_dir.joinpath(*pure_path.parts)
    current = model_dir
    for index, part in enumerate(pure_path.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            issue = f"locked model file is missing: {absolute_path}"
            return _file_fact(
                relative_path,
                absolute_path,
                evidence,
                status=ModelFileStatus.MISSING,
                issue=issue,
            )
        except OSError as error:
            issue = f"unable to inspect locked model path {current}: {error}"
            return _file_fact(
                relative_path,
                absolute_path,
                evidence,
                status=ModelFileStatus.UNAVAILABLE,
                issue=issue,
            )
        is_final = index == len(pure_path.parts) - 1
        if stat.S_ISLNK(metadata.st_mode) or (is_final and not stat.S_ISREG(metadata.st_mode)):
            issue = f"locked model path must be a regular file without symlinks: {current}"
            return _file_fact(
                relative_path,
                absolute_path,
                evidence,
                status=ModelFileStatus.NOT_REGULAR,
                actual_size_bytes=metadata.st_size,
                issue=issue,
            )
        if not is_final and not stat.S_ISDIR(metadata.st_mode):
            issue = f"locked model path has a non-directory parent: {current}"
            return _file_fact(
                relative_path,
                absolute_path,
                evidence,
                status=ModelFileStatus.NOT_REGULAR,
                issue=issue,
            )

    try:
        actual_size, actual_sha256 = _hash_regular_file(absolute_path)
    except OSError as error:
        issue = f"unable to hash locked model file {absolute_path}: {error}"
        return _file_fact(
            relative_path,
            absolute_path,
            evidence,
            status=ModelFileStatus.UNAVAILABLE,
            issue=issue,
        )
    if actual_size != evidence.size_bytes:
        issue = (
            f"model file size mismatch for {relative_path}: expected {evidence.size_bytes}, "
            f"found {actual_size}"
        )
        return _file_fact(
            relative_path,
            absolute_path,
            evidence,
            status=ModelFileStatus.SIZE_MISMATCH,
            actual_size_bytes=actual_size,
            actual_sha256=actual_sha256,
            issue=issue,
        )
    if actual_sha256 != evidence.sha256:
        issue = (
            f"model file SHA-256 mismatch for {relative_path}: expected {evidence.sha256}, "
            f"found {actual_sha256}"
        )
        return _file_fact(
            relative_path,
            absolute_path,
            evidence,
            status=ModelFileStatus.SHA256_MISMATCH,
            actual_size_bytes=actual_size,
            actual_sha256=actual_sha256,
            issue=issue,
        )
    return _file_fact(
        relative_path,
        absolute_path,
        evidence,
        status=ModelFileStatus.MATCH,
        actual_size_bytes=actual_size,
        actual_sha256=actual_sha256,
    )


def _enumerate_safetensors(model_dir: Path) -> tuple[set[str], list[str]]:
    """Enumerate regular safetensors without following any model-tree symlink."""
    safetensors: set[str] = set()
    issues: list[str] = []
    directories = [model_dir]
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError as error:
            issues.append(f"unable to enumerate model directory {directory}: {error}")
            continue
        for entry in entries:
            path = directory / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                issues.append(f"unable to inspect model path {path}: {error}")
                continue
            if stat.S_ISLNK(metadata.st_mode):
                issues.append(f"model tree must not contain symlinks: {path}")
            elif stat.S_ISDIR(metadata.st_mode):
                directories.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                if path.suffix.casefold() == ".safetensors":
                    safetensors.add(path.relative_to(model_dir).as_posix())
            else:
                issues.append(f"model tree contains a non-regular path: {path}")
    return safetensors, issues


def inspect_model_identity(
    model_lock: ModelIdentityLock | str | Path,
    *,
    model_dir: str | Path,
    repository_id: str,
    revision: str,
    parameter_count: int,
) -> ModelIdentityFacts:
    """Collect structured model identity facts without raising for a mismatch."""
    lock = (
        load_model_lock(model_lock) if not isinstance(model_lock, ModelIdentityLock) else model_lock
    )
    resolved_model_dir = Path(model_dir).expanduser().absolute()
    issues: list[str] = []

    if repository_id != lock.repository_id:
        issues.append(
            f"model repository mismatch: expected {lock.repository_id}, found {repository_id}"
        )
    if revision != lock.revision:
        issues.append(f"model revision mismatch: expected {lock.revision}, found {revision}")
    if isinstance(parameter_count, bool) or parameter_count != lock.parameter_count:
        issues.append(
            f"model parameter count mismatch: expected {lock.parameter_count}, "
            f"found {parameter_count}"
        )

    try:
        root_status = resolved_model_dir.lstat()
    except OSError as error:
        issues.append(f"model directory is unavailable: {resolved_model_dir}: {error}")
        root_status = None
    if root_status is not None:
        if stat.S_ISLNK(root_status.st_mode):
            issues.append(f"model directory must not be a symlink: {resolved_model_dir}")
        elif not stat.S_ISDIR(root_status.st_mode):
            issues.append(f"model path must be a directory: {resolved_model_dir}")

    file_facts = [
        _inspect_locked_file(resolved_model_dir, relative_path, evidence)
        for relative_path, evidence in sorted(lock.files.items())
    ]
    issues.extend(fact.issue for fact in file_facts if fact.issue is not None)

    actual_safetensors: set[str] = set()
    if root_status is not None and stat.S_ISDIR(root_status.st_mode):
        actual_safetensors, tree_issues = _enumerate_safetensors(resolved_model_dir)
        issues.extend(tree_issues)
    locked_safetensors = {
        path for path in lock.files if PurePosixPath(path).suffix.casefold() == ".safetensors"
    }
    extra_safetensors = sorted(actual_safetensors - locked_safetensors)
    if extra_safetensors:
        issues.append("unlocked safetensors files found: " + ", ".join(extra_safetensors))

    return ModelIdentityFacts(
        lock_path=lock.lock_path,
        model_dir=resolved_model_dir,
        expected_repository_id=lock.repository_id,
        actual_repository_id=repository_id,
        expected_revision=lock.revision,
        actual_revision=revision,
        expected_parameter_count=lock.parameter_count,
        actual_parameter_count=parameter_count,
        files=file_facts,
        extra_safetensors=extra_safetensors,
        matches_lock=not issues,
        issues=issues,
    )


def require_model_identity(
    model_lock: ModelIdentityLock | str | Path,
    *,
    model_dir: str | Path,
    repository_id: str,
    revision: str,
    parameter_count: int,
) -> ModelIdentityFacts:
    """Return verified formal-model facts or raise an error carrying those facts."""
    facts = inspect_model_identity(
        model_lock,
        model_dir=model_dir,
        repository_id=repository_id,
        revision=revision,
        parameter_count=parameter_count,
    )
    if not facts.matches_lock:
        raise ModelIdentityError(facts)
    return facts
