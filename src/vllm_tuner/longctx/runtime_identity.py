"""Fail-closed attestation of the vLLM runtime used by long-context experiments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import platform
import subprocess
from collections.abc import Mapping
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

KNOWN_LEGACY_PATCHED_SCHEDULER_SHA256 = (
    "ed1b8dc7816a48b69710631e67d929f6d4c3870ce868f422cd820389bd08731c"
)
SCHEDULER_SOURCE_PATH = "vllm/v1/core/sched/scheduler.py"
_SHA256_LENGTH = 64

RUNTIME_ENVIRONMENT_FIELDS = (
    "python",
    "torch",
    "cuda",
    "transformers",
    "flashinfer_python",
    "nvidia_driver",
    "gpu",
    "gpu_memory_mib",
    "compute_capability",
)

RuntimeValue = str | int


class RuntimeSourceStatus(str, Enum):
    """Comparison outcome for one locked vLLM source file."""

    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING = "missing"
    LEGACY_PATCHED = "legacy_patched"


class ExpectedRuntimeEnvironment(BaseModel):
    """Strict runtime and GPU identity expected by the reproduction lock."""

    python: str
    torch: str
    cuda: str
    transformers: str
    flashinfer_python: str
    nvidia_driver: str
    gpu: str
    gpu_memory_mib: int
    compute_capability: str

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RuntimeEnvironmentSnapshot(BaseModel):
    """Observed runtime values, with unavailable facts represented by ``None``."""

    python: Optional[str] = None
    torch: Optional[str] = None
    cuda: Optional[str] = None
    transformers: Optional[str] = None
    flashinfer_python: Optional[str] = None
    nvidia_driver: Optional[str] = None
    gpu: Optional[str] = None
    gpu_memory_mib: Optional[int] = None
    compute_capability: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class VLLMRuntimeLock(BaseModel):
    """Expected upstream identity parsed from the YAML runtime lock."""

    lock_path: Path
    expected_version: str
    upstream_commit: str
    wheel_record_sha256: str
    source_files: dict[str, str]
    runtime_environment: ExpectedRuntimeEnvironment
    legacy_patched_scheduler_sha256: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeSourceFileFact(BaseModel):
    """Expected and observed identity for one installed vLLM source file."""

    lock_path: str
    package_relative_path: str
    absolute_path: Optional[Path] = None
    expected_sha256: str
    actual_sha256: Optional[str] = None
    status: RuntimeSourceStatus
    issue: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeEnvironmentFieldFact(BaseModel):
    """Expected/actual comparison for one locked runtime environment field."""

    name: str
    expected: RuntimeValue
    actual: Optional[RuntimeValue] = None
    matches: bool
    issue: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeWheelRecordFact(BaseModel):
    """Expected/actual comparison for the installed vLLM wheel RECORD file."""

    expected_sha256: str
    actual_sha256: Optional[str] = None
    record_path: Optional[Path] = None
    matches: bool
    issue: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeIdentityFacts(BaseModel):
    """Structured evidence explaining whether the active vLLM is clean upstream."""

    lock_path: Path
    upstream_commit: str
    expected_version: str
    actual_version: Optional[str]
    package_dir: Optional[Path]
    wheel_record: RuntimeWheelRecordFact
    environment: dict[str, RuntimeEnvironmentFieldFact]
    source_files: list[RuntimeSourceFileFact] = Field(default_factory=list)
    legacy_patched_scheduler: bool = False
    matches_lock: bool
    issues: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeIdentityError(ValueError):
    """Raised when a runtime cannot be proven to match the clean upstream lock."""

    def __init__(self, facts: RuntimeIdentityFacts) -> None:
        self.facts = facts
        detail = "; ".join(facts.issues) if facts.issues else "unknown identity mismatch"
        super().__init__(f"vLLM runtime identity check failed: {detail}")


def _mapping(value: object, field: str) -> Mapping[object, object]:
    """Return a mapping or reject an invalid lock field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"runtime lock field '{field}' must be a mapping")
    return value


def _nonempty_string(value: object, field: str) -> str:
    """Return a stripped non-empty string or reject an invalid lock field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime lock field '{field}' must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    """Return a positive integer or reject an invalid lock field."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"runtime lock field '{field}' must be a positive integer")
    return value


def _sha256(value: object, field: str) -> str:
    """Return a normalized SHA-256 digest or reject malformed lock evidence."""
    digest = _nonempty_string(value, field).lower()
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"runtime lock field '{field}' must be a 64-character SHA-256")
    return digest


def _aliased_string(mapping: Mapping[object, object], aliases: tuple[str, ...], field: str) -> str:
    """Read one logical string field while rejecting conflicting aliases."""
    values = [
        _nonempty_string(mapping[alias], f"{field}.{alias}")
        for alias in aliases
        if alias in mapping and mapping[alias] is not None
    ]
    if not values:
        aliases_text = ", ".join(aliases)
        raise ValueError(f"runtime lock field '{field}' requires one of: {aliases_text}")
    if len(set(values)) != 1:
        raise ValueError(f"runtime lock field '{field}' has conflicting aliases")
    return values[0]


def _validate_source_path(value: object) -> str:
    """Validate and normalize a source path relative to the installed package."""
    source_path = _nonempty_string(value, "vllm.source_files path")
    if "\\" in source_path:
        raise ValueError(f"runtime lock source path must use POSIX separators: {source_path}")
    pure_path = PurePosixPath(source_path)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError(f"runtime lock source path is not safely relative: {source_path}")
    if pure_path.parts[0] == "vllm" and len(pure_path.parts) == 1:
        raise ValueError("runtime lock source path must identify a file below vllm/")
    return pure_path.as_posix()


def _source_hash(value: object, field: str) -> str:
    """Read either a direct digest or a mapping containing a sha256 field."""
    if isinstance(value, Mapping):
        if "sha256" not in value:
            raise ValueError(f"runtime lock field '{field}' must contain sha256")
        value = value["sha256"]
    return _sha256(value, field)


def load_runtime_lock(lock_path: str | Path) -> VLLMRuntimeLock:
    """Load the expected vLLM version, commit, and critical source hashes from YAML."""
    resolved_lock_path = Path(lock_path).expanduser().resolve()
    try:
        payload = yaml.safe_load(resolved_lock_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"unable to read vLLM runtime lock {resolved_lock_path}: {error}"
        ) from error

    root = _mapping(payload, "root")
    vllm = _mapping(root.get("vllm"), "vllm")
    expected_version = _nonempty_string(vllm.get("version"), "vllm.version")
    upstream_commit = _aliased_string(
        vllm,
        ("upstream_commit", "tag_commit", "commit"),
        "vllm upstream commit",
    )

    wheel_record_sha256 = _sha256(
        vllm.get("wheel_record_sha256"),
        "vllm.wheel_record_sha256",
    )
    runtime = _mapping(root.get("runtime"), "runtime")
    runtime_environment = ExpectedRuntimeEnvironment(
        python=_nonempty_string(runtime.get("python"), "runtime.python"),
        torch=_nonempty_string(runtime.get("torch"), "runtime.torch"),
        cuda=_nonempty_string(runtime.get("cuda"), "runtime.cuda"),
        transformers=_nonempty_string(runtime.get("transformers"), "runtime.transformers"),
        flashinfer_python=_nonempty_string(
            runtime.get("flashinfer_python"), "runtime.flashinfer_python"
        ),
        nvidia_driver=_nonempty_string(runtime.get("nvidia_driver"), "runtime.nvidia_driver"),
        gpu=_nonempty_string(runtime.get("gpu"), "runtime.gpu"),
        gpu_memory_mib=_positive_int(runtime.get("gpu_memory_mib"), "runtime.gpu_memory_mib"),
        compute_capability=_nonempty_string(
            runtime.get("compute_capability"), "runtime.compute_capability"
        ),
    )

    raw_source_files = _mapping(vllm.get("source_files"), "vllm.source_files")
    if not raw_source_files:
        raise ValueError("runtime lock field 'vllm.source_files' must not be empty")
    source_files: dict[str, str] = {}
    for raw_path, raw_digest in raw_source_files.items():
        source_path = _validate_source_path(raw_path)
        if source_path in source_files:
            raise ValueError(f"duplicate normalized runtime lock source path: {source_path}")
        source_files[source_path] = _source_hash(
            raw_digest,
            f"vllm.source_files.{source_path}",
        )

    legacy_hash: Optional[str] = None
    raw_patch = vllm.get("slotune_patch")
    if raw_patch is not None:
        patch = _mapping(raw_patch, "vllm.slotune_patch")
        raw_legacy_hash = patch.get("patched_scheduler_sha256")
        if raw_legacy_hash is not None:
            legacy_hash = _sha256(
                raw_legacy_hash,
                "vllm.slotune_patch.patched_scheduler_sha256",
            )

    return VLLMRuntimeLock(
        lock_path=resolved_lock_path,
        expected_version=expected_version,
        upstream_commit=upstream_commit,
        wheel_record_sha256=wheel_record_sha256,
        source_files=source_files,
        runtime_environment=runtime_environment,
        legacy_patched_scheduler_sha256=legacy_hash,
    )


def _package_relative_path(lock_source_path: str) -> str:
    """Convert a lock path such as vllm/v1/request.py to a package-root path."""
    parts = PurePosixPath(lock_source_path).parts
    if parts[0] == "vllm":
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix()


def _hash_file(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_vllm_wheel_record() -> tuple[Optional[Path], Optional[str]]:
    """Locate the installed vLLM distribution RECORD file."""
    try:
        distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None, "vLLM distribution RECORD is unavailable because vLLM is not installed"
    except Exception as error:  # pragma: no cover - defensive metadata backend boundary
        return None, f"unable to inspect installed vLLM distribution metadata: {error}"

    record_entries = [
        entry
        for entry in distribution.files or ()
        if entry.name == "RECORD" and entry.parent.name.endswith(".dist-info")
    ]
    if len(record_entries) != 1:
        return None, (
            "unable to identify exactly one vLLM distribution RECORD file; "
            f"found {len(record_entries)}"
        )
    record_path = Path(str(distribution.locate_file(record_entries[0]))).resolve()
    if not record_path.is_file():
        return record_path, f"vLLM distribution RECORD file is missing: {record_path}"
    return record_path, None


def _inspect_wheel_record(
    expected_sha256: str,
    *,
    record_path: str | Path | None,
    supplied_sha256: Optional[str],
) -> RuntimeWheelRecordFact:
    """Compare the current vLLM wheel RECORD against its locked digest."""
    if record_path is not None and supplied_sha256 is not None:
        raise ValueError("provide only one of wheel_record_path and wheel_record_sha256")

    resolved_record_path: Optional[Path] = None
    actual_sha256: Optional[str] = None
    discovery_issue: Optional[str] = None
    if supplied_sha256 is not None:
        actual_sha256 = _sha256(supplied_sha256, "injected wheel_record_sha256")
    else:
        if record_path is None:
            resolved_record_path, discovery_issue = _discover_vllm_wheel_record()
        else:
            resolved_record_path = Path(record_path).expanduser().resolve()
        if discovery_issue is None and resolved_record_path is not None:
            if not resolved_record_path.is_file():
                discovery_issue = (
                    f"vLLM distribution RECORD file is missing: {resolved_record_path}"
                )
            else:
                try:
                    actual_sha256 = _hash_file(resolved_record_path)
                except OSError as error:
                    discovery_issue = (
                        f"unable to hash vLLM distribution RECORD {resolved_record_path}: {error}"
                    )

    if actual_sha256 is None:
        issue = discovery_issue or "vLLM distribution RECORD SHA-256 is unavailable"
        return RuntimeWheelRecordFact(
            expected_sha256=expected_sha256,
            record_path=resolved_record_path,
            matches=False,
            issue=issue,
        )
    if actual_sha256 != expected_sha256:
        issue = (
            "vLLM wheel RECORD SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
        return RuntimeWheelRecordFact(
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            record_path=resolved_record_path,
            matches=False,
            issue=issue,
        )
    return RuntimeWheelRecordFact(
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        record_path=resolved_record_path,
        matches=True,
    )


def _discover_vllm_version() -> tuple[Optional[str], Optional[str]]:
    """Return the current interpreter's installed vLLM distribution version."""
    try:
        return importlib.metadata.version("vllm"), None
    except importlib.metadata.PackageNotFoundError:
        return None, "vLLM distribution is not installed in the current interpreter"
    except Exception as error:  # pragma: no cover - defensive metadata backend boundary
        return None, f"unable to read installed vLLM version: {error}"


def _discover_vllm_package_dir() -> tuple[Optional[Path], Optional[str]]:
    """Locate the current interpreter's importable vllm package without importing it."""
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, AttributeError, ValueError) as error:
        return None, f"unable to locate importable vllm package: {error}"
    if spec is None:
        return None, "vllm package is not importable in the current interpreter"
    if spec.submodule_search_locations:
        first_location = next(iter(spec.submodule_search_locations), None)
        if first_location is not None:
            return Path(first_location).resolve(), None
    if spec.origin:
        return Path(spec.origin).resolve().parent, None
    return None, "importable vllm package has no filesystem location"


def _installed_distribution_version(
    distribution_name: str,
    fact_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """Read one installed package version without importing the package."""
    try:
        return importlib.metadata.version(distribution_name), None
    except importlib.metadata.PackageNotFoundError:
        return None, f"{fact_name} distribution is not installed"
    except Exception as error:  # pragma: no cover - defensive metadata backend boundary
        return None, f"unable to read installed {fact_name} version: {error}"


def _torch_cuda_version() -> tuple[Optional[str], Optional[str]]:
    """Read the CUDA version against which the active Torch package was built."""
    try:
        torch_module = importlib.import_module("torch")
        torch_version = getattr(torch_module, "version", None)
        cuda_version = getattr(torch_version, "cuda", None)
    except Exception as error:  # pragma: no cover - compiled package import boundary
        return None, f"unable to import torch for its CUDA version: {error}"
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        return None, "torch.version.cuda is unavailable"
    return cuda_version.strip(), None


def _nvidia_smi_facts() -> tuple[dict[str, RuntimeValue], dict[str, str]]:
    """Collect the first visible GPU's locked identity fields through nvidia-smi."""
    gpu_fields = ("nvidia_driver", "gpu", "gpu_memory_mib", "compute_capability")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        issue = f"unable to collect GPU identity with nvidia-smi: {error}"
        return {}, {name: issue for name in gpu_fields}

    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        issue = "nvidia-smi returned no GPU identity rows"
        return {}, {name: issue for name in gpu_fields}
    columns = [value.strip() for value in rows[0].split(",", maxsplit=3)]
    if len(columns) != len(gpu_fields):
        issue = f"nvidia-smi returned an invalid GPU identity row: {rows[0]!r}"
        return {}, {name: issue for name in gpu_fields}

    values: dict[str, RuntimeValue] = {}
    issues: dict[str, str] = {}
    for name, raw_value in (
        ("nvidia_driver", columns[0]),
        ("gpu", columns[1]),
        ("compute_capability", columns[3]),
    ):
        if not raw_value or raw_value.upper() == "N/A":
            issues[name] = f"nvidia-smi did not report {name}"
        else:
            values[name] = raw_value

    try:
        values["gpu_memory_mib"] = int(columns[2])
    except ValueError:
        issues["gpu_memory_mib"] = f"nvidia-smi returned invalid GPU memory: {columns[2]!r}"
    return values, issues


def _collect_runtime_environment() -> tuple[RuntimeEnvironmentSnapshot, dict[str, str]]:
    """Collect package, CUDA, Python, driver, and GPU identity for this interpreter."""
    values: dict[str, object] = {"python": platform.python_version()}
    issues: dict[str, str] = {}
    for field_name, distribution_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("flashinfer_python", "flashinfer-python"),
    ):
        value, issue = _installed_distribution_version(distribution_name, field_name)
        values[field_name] = value
        if issue is not None:
            issues[field_name] = issue

    cuda_version, cuda_issue = _torch_cuda_version()
    values["cuda"] = cuda_version
    if cuda_issue is not None:
        issues["cuda"] = cuda_issue

    gpu_values, gpu_issues = _nvidia_smi_facts()
    values.update(gpu_values)
    issues.update(gpu_issues)
    return RuntimeEnvironmentSnapshot.model_validate(values), issues


def _coerce_environment_snapshot(
    environment_facts: RuntimeEnvironmentSnapshot | Mapping[str, RuntimeValue | None],
) -> RuntimeEnvironmentSnapshot:
    """Normalize injected environment facts for deterministic CPU-only tests."""
    if isinstance(environment_facts, RuntimeEnvironmentSnapshot):
        return environment_facts
    return RuntimeEnvironmentSnapshot.model_validate(dict(environment_facts))


def _compare_runtime_environment(
    expected: ExpectedRuntimeEnvironment,
    actual: RuntimeEnvironmentSnapshot,
    collection_issues: Mapping[str, str],
) -> dict[str, RuntimeEnvironmentFieldFact]:
    """Build one fail-closed expected/actual fact per locked environment field."""
    expected_values = expected.model_dump()
    actual_values = actual.model_dump()
    facts: dict[str, RuntimeEnvironmentFieldFact] = {}
    for name in RUNTIME_ENVIRONMENT_FIELDS:
        expected_value: RuntimeValue = expected_values[name]
        actual_value: Optional[RuntimeValue] = actual_values[name]
        issue: Optional[str] = None
        if actual_value is None:
            detail = collection_issues.get(name)
            issue = (
                f"runtime environment field {name!r} is unavailable; expected {expected_value!r}"
            )
            if detail is not None:
                issue = f"{issue}: {detail}"
        elif actual_value != expected_value:
            issue = (
                f"runtime environment mismatch for {name}: "
                f"expected {expected_value!r}, found {actual_value!r}"
            )
        facts[name] = RuntimeEnvironmentFieldFact(
            name=name,
            expected=expected_value,
            actual=actual_value,
            matches=issue is None,
            issue=issue,
        )
    return facts


def _legacy_scheduler_hashes(runtime_lock: VLLMRuntimeLock) -> frozenset[str]:
    """Return all explicitly known Legacy Scheduler digests."""
    hashes = {KNOWN_LEGACY_PATCHED_SCHEDULER_SHA256}
    if runtime_lock.legacy_patched_scheduler_sha256 is not None:
        hashes.add(runtime_lock.legacy_patched_scheduler_sha256)
    return frozenset(hashes)


def _is_scheduler_source(lock_source_path: str) -> bool:
    """Return whether a locked source path is the v1 Scheduler implementation."""
    return _package_relative_path(lock_source_path) == _package_relative_path(SCHEDULER_SOURCE_PATH)


def _inspect_source_file(
    runtime_lock: VLLMRuntimeLock,
    lock_source_path: str,
    expected_sha256: str,
    package_dir: Optional[Path],
) -> RuntimeSourceFileFact:
    """Compare one critical source file against upstream and Legacy digests."""
    package_relative_path = _package_relative_path(lock_source_path)
    if package_dir is None:
        issue = f"cannot inspect vLLM source file without package directory: {lock_source_path}"
        return RuntimeSourceFileFact(
            lock_path=lock_source_path,
            package_relative_path=package_relative_path,
            expected_sha256=expected_sha256,
            status=RuntimeSourceStatus.MISSING,
            issue=issue,
        )

    absolute_path = package_dir.joinpath(*PurePosixPath(package_relative_path).parts)
    if not absolute_path.is_file():
        issue = f"locked vLLM source file is missing: {absolute_path}"
        return RuntimeSourceFileFact(
            lock_path=lock_source_path,
            package_relative_path=package_relative_path,
            absolute_path=absolute_path,
            expected_sha256=expected_sha256,
            status=RuntimeSourceStatus.MISSING,
            issue=issue,
        )

    try:
        actual_sha256 = _hash_file(absolute_path)
    except OSError as error:
        issue = f"unable to hash locked vLLM source file {absolute_path}: {error}"
        return RuntimeSourceFileFact(
            lock_path=lock_source_path,
            package_relative_path=package_relative_path,
            absolute_path=absolute_path,
            expected_sha256=expected_sha256,
            status=RuntimeSourceStatus.MISSING,
            issue=issue,
        )

    if _is_scheduler_source(lock_source_path) and actual_sha256 in _legacy_scheduler_hashes(
        runtime_lock
    ):
        issue = (
            "legacy patched scheduler detected at "
            f"{absolute_path}: SHA-256 {actual_sha256}; v5 requires clean upstream vLLM"
        )
        return RuntimeSourceFileFact(
            lock_path=lock_source_path,
            package_relative_path=package_relative_path,
            absolute_path=absolute_path,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            status=RuntimeSourceStatus.LEGACY_PATCHED,
            issue=issue,
        )

    if actual_sha256 != expected_sha256:
        issue = (
            f"vLLM source hash mismatch for {absolute_path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
        return RuntimeSourceFileFact(
            lock_path=lock_source_path,
            package_relative_path=package_relative_path,
            absolute_path=absolute_path,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            status=RuntimeSourceStatus.MISMATCH,
            issue=issue,
        )

    return RuntimeSourceFileFact(
        lock_path=lock_source_path,
        package_relative_path=package_relative_path,
        absolute_path=absolute_path,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        status=RuntimeSourceStatus.MATCH,
    )


def inspect_runtime_identity(
    runtime_lock: VLLMRuntimeLock | str | Path,
    *,
    package_dir: str | Path | None = None,
    version: Optional[str] = None,
    environment_facts: RuntimeEnvironmentSnapshot | Mapping[str, RuntimeValue | None] | None = None,
    wheel_record_path: str | Path | None = None,
    wheel_record_sha256: Optional[str] = None,
) -> RuntimeIdentityFacts:
    """Collect structured facts without raising for an installed-runtime mismatch."""
    lock = (
        load_runtime_lock(runtime_lock)
        if not isinstance(runtime_lock, VLLMRuntimeLock)
        else runtime_lock
    )
    issues: list[str] = []

    actual_version = version
    if actual_version is None:
        actual_version, version_issue = _discover_vllm_version()
        if version_issue is not None:
            issues.append(version_issue)
    if actual_version is None:
        issues.append(f"vLLM version is unavailable; expected {lock.expected_version}")
    elif actual_version != lock.expected_version:
        issues.append(
            f"vLLM version mismatch: expected {lock.expected_version}, found {actual_version}"
        )

    wheel_record = _inspect_wheel_record(
        lock.wheel_record_sha256,
        record_path=wheel_record_path,
        supplied_sha256=wheel_record_sha256,
    )
    if wheel_record.issue is not None:
        issues.append(wheel_record.issue)

    if environment_facts is None:
        environment_snapshot, environment_collection_issues = _collect_runtime_environment()
    else:
        environment_snapshot = _coerce_environment_snapshot(environment_facts)
        environment_collection_issues = {}
    environment = _compare_runtime_environment(
        lock.runtime_environment,
        environment_snapshot,
        environment_collection_issues,
    )
    issues.extend(fact.issue for fact in environment.values() if fact.issue is not None)

    resolved_package_dir: Optional[Path]
    if package_dir is None:
        resolved_package_dir, package_issue = _discover_vllm_package_dir()
        if package_issue is not None:
            issues.append(package_issue)
    else:
        resolved_package_dir = Path(package_dir).expanduser().resolve()
    if resolved_package_dir is not None and not resolved_package_dir.is_dir():
        issues.append(f"vLLM package directory does not exist: {resolved_package_dir}")

    source_facts = [
        _inspect_source_file(lock, source_path, expected_hash, resolved_package_dir)
        for source_path, expected_hash in sorted(lock.source_files.items())
    ]
    issues.extend(fact.issue for fact in source_facts if fact.issue is not None)
    legacy_patched_scheduler = any(
        fact.status == RuntimeSourceStatus.LEGACY_PATCHED for fact in source_facts
    )

    return RuntimeIdentityFacts(
        lock_path=lock.lock_path,
        upstream_commit=lock.upstream_commit,
        expected_version=lock.expected_version,
        actual_version=actual_version,
        package_dir=resolved_package_dir,
        wheel_record=wheel_record,
        environment=environment,
        source_files=source_facts,
        legacy_patched_scheduler=legacy_patched_scheduler,
        matches_lock=not issues,
        issues=issues,
    )


def require_upstream_runtime(
    runtime_lock: VLLMRuntimeLock | str | Path,
    *,
    package_dir: str | Path | None = None,
    version: Optional[str] = None,
    environment_facts: RuntimeEnvironmentSnapshot | Mapping[str, RuntimeValue | None] | None = None,
    wheel_record_path: str | Path | None = None,
    wheel_record_sha256: Optional[str] = None,
) -> RuntimeIdentityFacts:
    """Return attested runtime facts or fail closed with a descriptive ValueError."""
    facts = inspect_runtime_identity(
        runtime_lock,
        package_dir=package_dir,
        version=version,
        environment_facts=environment_facts,
        wheel_record_path=wheel_record_path,
        wheel_record_sha256=wheel_record_sha256,
    )
    if not facts.matches_lock:
        raise RuntimeIdentityError(facts)
    return facts
