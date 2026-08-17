"""Unit tests for fail-closed vLLM runtime identity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vllm_tuner.longctx.runtime_identity import (
    RuntimeIdentityError,
    RuntimeIdentityFacts,
    RuntimeSourceStatus,
    inspect_runtime_identity as _inspect_runtime_identity,
    load_runtime_lock,
    require_upstream_runtime as _require_upstream_runtime,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_RECORD_PAYLOAD = b"vllm wheel record evidence\n"
_RUNTIME_ENVIRONMENT: dict[str, str | int] = {
    "python": "3.12.3",
    "torch": "2.9.1+cu130",
    "cuda": "13.0",
    "transformers": "4.57.6",
    "flashinfer_python": "0.6.3",
    "nvidia_driver": "595.71.05",
    "gpu": "NVIDIA GeForce RTX 5090",
    "gpu_memory_mib": 32607,
    "compute_capability": "12.0",
}


def _write_record(tmp_path: Path) -> Path:
    record_path = tmp_path / "vllm-0.16.0.dist-info" / "RECORD"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(_RECORD_PAYLOAD)
    return record_path


def inspect_runtime_identity(
    runtime_lock: Path,
    *,
    package_dir: Path,
    version: str,
) -> RuntimeIdentityFacts:
    """Inspect a fake runtime with all external environment facts injected."""
    return _inspect_runtime_identity(
        runtime_lock,
        package_dir=package_dir,
        version=version,
        environment_facts=_RUNTIME_ENVIRONMENT,
        wheel_record_path=_write_record(runtime_lock.parent),
    )


def require_upstream_runtime(
    runtime_lock: Path,
    *,
    package_dir: Path,
    version: str,
) -> RuntimeIdentityFacts:
    """Require a fake runtime with all external environment facts injected."""
    return _require_upstream_runtime(
        runtime_lock,
        package_dir=package_dir,
        version=version,
        environment_facts=_RUNTIME_ENVIRONMENT,
        wheel_record_path=_write_record(runtime_lock.parent),
    )


def _write_lock(
    tmp_path: Path,
    source_files: dict[str, str],
    *,
    patched_scheduler_sha256: str | None = None,
) -> Path:
    source_lines = "\n".join(f'    {path}: "{digest}"' for path, digest in source_files.items())
    patch_lines = ""
    if patched_scheduler_sha256 is not None:
        patch_lines = (
            "\n  slotune_patch:\n" f'    patched_scheduler_sha256: "{patched_scheduler_sha256}"\n'
        )
    _write_record(tmp_path)
    lock_path = tmp_path / "upstream.lock.yaml"
    lock_path.write_text(
        "vllm:\n"
        '  version: "0.16.0"\n'
        '  tag_commit: "89a77b10846fd96273cce78d86d2556ea582d26e"\n'
        f'  wheel_record_sha256: "{_sha256(_RECORD_PAYLOAD)}"\n'
        "  source_files:\n"
        f"{source_lines}"
        f"{patch_lines}"
        "\nruntime:\n"
        f'  python: "{_RUNTIME_ENVIRONMENT["python"]}"\n'
        f'  torch: "{_RUNTIME_ENVIRONMENT["torch"]}"\n'
        f'  cuda: "{_RUNTIME_ENVIRONMENT["cuda"]}"\n'
        f'  transformers: "{_RUNTIME_ENVIRONMENT["transformers"]}"\n'
        f'  flashinfer_python: "{_RUNTIME_ENVIRONMENT["flashinfer_python"]}"\n'
        f'  nvidia_driver: "{_RUNTIME_ENVIRONMENT["nvidia_driver"]}"\n'
        f'  gpu: "{_RUNTIME_ENVIRONMENT["gpu"]}"\n'
        f'  gpu_memory_mib: {_RUNTIME_ENVIRONMENT["gpu_memory_mib"]}\n'
        f'  compute_capability: "{_RUNTIME_ENVIRONMENT["compute_capability"]}"\n',
        encoding="utf-8",
    )
    return lock_path


def _write_package_file(package_dir: Path, relative_path: str, payload: bytes) -> Path:
    path = package_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_load_runtime_lock_reads_upstream_identity(tmp_path: Path) -> None:
    payload = b"upstream request source\n"
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/request.py": _sha256(payload)},
    )

    runtime_lock = load_runtime_lock(lock_path)

    assert runtime_lock.expected_version == "0.16.0"
    assert runtime_lock.upstream_commit == "89a77b10846fd96273cce78d86d2556ea582d26e"
    assert runtime_lock.source_files == {"vllm/v1/request.py": _sha256(payload)}
    assert runtime_lock.wheel_record_sha256 == _sha256(_RECORD_PAYLOAD)
    assert runtime_lock.runtime_environment.model_dump() == _RUNTIME_ENVIRONMENT


def test_inspect_runtime_identity_accepts_injected_clean_package(tmp_path: Path) -> None:
    package_dir = tmp_path / "site-packages" / "vllm"
    request_payload = b"clean request source\n"
    scheduler_payload = b"clean scheduler source\n"
    request_path = _write_package_file(package_dir, "v1/request.py", request_payload)
    scheduler_path = _write_package_file(
        package_dir,
        "v1/core/sched/scheduler.py",
        scheduler_payload,
    )
    lock_path = _write_lock(
        tmp_path,
        {
            "vllm/v1/request.py": _sha256(request_payload),
            "vllm/v1/core/sched/scheduler.py": _sha256(scheduler_payload),
        },
    )

    facts = inspect_runtime_identity(lock_path, package_dir=package_dir, version="0.16.0")

    assert facts.matches_lock is True
    assert facts.issues == []
    assert facts.wheel_record.matches is True
    assert all(fact.matches for fact in facts.environment.values())
    assert facts.package_dir == package_dir.resolve()
    assert facts.upstream_commit == "89a77b10846fd96273cce78d86d2556ea582d26e"
    assert {fact.absolute_path for fact in facts.source_files} == {request_path, scheduler_path}
    assert {fact.status for fact in facts.source_files} == {RuntimeSourceStatus.MATCH}
    assert (
        require_upstream_runtime(
            lock_path,
            package_dir=package_dir,
            version="0.16.0",
        )
        == facts
    )


def test_require_upstream_runtime_rejects_record_and_environment_mismatch(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "vllm"
    source_payload = b"clean request source\n"
    _write_package_file(package_dir, "v1/request.py", source_payload)
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/request.py": _sha256(source_payload)},
    )
    record_path = _write_record(tmp_path)
    record_path.write_bytes(b"tampered wheel RECORD\n")
    mismatched_environment: dict[str, str | int] = {
        **_RUNTIME_ENVIRONMENT,
        "python": "3.11.0",
        "gpu_memory_mib": 24576,
    }

    facts = _inspect_runtime_identity(
        lock_path,
        package_dir=package_dir,
        version="0.16.0",
        environment_facts=mismatched_environment,
        wheel_record_path=record_path,
    )

    assert facts.matches_lock is False
    assert facts.wheel_record.matches is False
    assert facts.wheel_record.record_path == record_path.resolve()
    assert facts.environment["python"].matches is False
    assert facts.environment["python"].expected == "3.12.3"
    assert facts.environment["python"].actual == "3.11.0"
    assert facts.environment["gpu_memory_mib"].matches is False
    assert facts.environment["transformers"].matches is True
    assert any("wheel RECORD SHA-256 mismatch" in issue for issue in facts.issues)
    assert any("runtime environment mismatch for python" in issue for issue in facts.issues)
    with pytest.raises(RuntimeIdentityError, match="wheel RECORD SHA-256 mismatch"):
        _require_upstream_runtime(
            lock_path,
            package_dir=package_dir,
            version="0.16.0",
            environment_facts=mismatched_environment,
            wheel_record_path=record_path,
        )


def test_require_upstream_runtime_rejects_version_and_source_mismatch(tmp_path: Path) -> None:
    package_dir = tmp_path / "vllm"
    expected_payload = b"expected\n"
    _write_package_file(package_dir, "v1/request.py", b"modified\n")
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/request.py": _sha256(expected_payload)},
    )

    facts = inspect_runtime_identity(lock_path, package_dir=package_dir, version="0.15.0")

    assert facts.matches_lock is False
    assert facts.source_files[0].status == RuntimeSourceStatus.MISMATCH
    assert any("version mismatch" in issue for issue in facts.issues)
    assert any("source hash mismatch" in issue for issue in facts.issues)
    with pytest.raises(RuntimeIdentityError, match="version mismatch") as error_info:
        require_upstream_runtime(lock_path, package_dir=package_dir, version="0.15.0")
    assert error_info.value.facts == facts


def test_require_upstream_runtime_explicitly_rejects_legacy_scheduler(tmp_path: Path) -> None:
    package_dir = tmp_path / "vllm"
    upstream_payload = b"upstream scheduler\n"
    legacy_payload = b"legacy patched scheduler\n"
    _write_package_file(package_dir, "v1/core/sched/scheduler.py", legacy_payload)
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/core/sched/scheduler.py": _sha256(upstream_payload)},
        patched_scheduler_sha256=_sha256(legacy_payload),
    )

    facts = inspect_runtime_identity(lock_path, package_dir=package_dir, version="0.16.0")

    assert facts.matches_lock is False
    assert facts.legacy_patched_scheduler is True
    assert facts.source_files[0].status == RuntimeSourceStatus.LEGACY_PATCHED
    assert "legacy patched scheduler detected" in facts.issues[0]
    with pytest.raises(ValueError, match="legacy patched scheduler detected"):
        require_upstream_runtime(lock_path, package_dir=package_dir, version="0.16.0")


def test_inspect_runtime_identity_reports_missing_locked_source(tmp_path: Path) -> None:
    package_dir = tmp_path / "vllm"
    package_dir.mkdir()
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/request.py": _sha256(b"expected\n")},
    )

    facts = inspect_runtime_identity(lock_path, package_dir=package_dir, version="0.16.0")

    assert facts.matches_lock is False
    assert facts.source_files[0].status == RuntimeSourceStatus.MISSING
    assert "source file is missing" in facts.issues[0]


def test_load_runtime_lock_requires_wheel_record_sha256(tmp_path: Path) -> None:
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/request.py": _sha256(b"expected\n")},
    )
    payload = lock_path.read_text(encoding="utf-8")
    payload = payload.replace(
        f'  wheel_record_sha256: "{_sha256(_RECORD_PAYLOAD)}"\n',
        "",
    )
    lock_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="vllm.wheel_record_sha256"):
        load_runtime_lock(lock_path)


def test_load_runtime_lock_requires_complete_runtime_mapping(tmp_path: Path) -> None:
    lock_path = _write_lock(
        tmp_path,
        {"vllm/v1/request.py": _sha256(b"expected\n")},
    )
    payload = lock_path.read_text(encoding="utf-8").partition("\nruntime:\n")[0]
    lock_path.write_text(f"{payload}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="field 'runtime' must be a mapping"):
        load_runtime_lock(lock_path)


@pytest.mark.parametrize(
    "source_path,digest,error",
    [
        ("../scheduler.py", "0" * 64, "not safely relative"),
        ("vllm/v1/request.py", "not-a-digest", "64-character SHA-256"),
    ],
)
def test_load_runtime_lock_rejects_unsafe_or_unverifiable_source(
    tmp_path: Path,
    source_path: str,
    digest: str,
    error: str,
) -> None:
    lock_path = _write_lock(tmp_path, {source_path: digest})

    with pytest.raises(ValueError, match=error):
        load_runtime_lock(lock_path)
