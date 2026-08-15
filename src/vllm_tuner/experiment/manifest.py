"""Environment fingerprinting, checksums, and safe resume validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import EnvironmentFingerprint, ExperimentSpec, ModelWeightFingerprint

TRACKED_PACKAGES = (
    "vllm",
    "torch",
    "flashinfer-python",
    "transformers",
    "optuna",
    "numpy",
    "nvidia-ml-py",
)

LOCAL_MODEL_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".gguf"})
LOCAL_MODEL_METADATA_FILES = (
    "config.json",
    "configuration.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "quantize_config.json",
    "quantization_config.json",
    "quant_config.json",
    "adapter_config.json",
)
LOCAL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using stable serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _local_model_weight_paths(model_path: Path) -> list[Path]:
    """Return local weight files in a path-independent deterministic order."""
    if model_path.is_file():
        return [model_path] if model_path.suffix.lower() in LOCAL_MODEL_WEIGHT_SUFFIXES else []
    if not model_path.is_dir():
        return []
    return sorted(
        (
            path
            for path in model_path.rglob("*")
            if path.is_file() and path.suffix.lower() in LOCAL_MODEL_WEIGHT_SUFFIXES
        ),
        key=lambda path: path.relative_to(model_path).as_posix(),
    )


def fingerprint_local_model_weights(model: str | Path) -> list[ModelWeightFingerprint]:
    """Hash every local model weight file without embedding its absolute root."""
    model_path = Path(model)
    weight_paths = _local_model_weight_paths(model_path)
    fingerprints: list[ModelWeightFingerprint] = []
    for weight_path in weight_paths:
        relative_path = (
            weight_path.name
            if model_path.is_file()
            else weight_path.relative_to(model_path).as_posix()
        )
        fingerprints.append(
            ModelWeightFingerprint(
                path=relative_path,
                size_bytes=weight_path.stat().st_size,
                sha256=sha256_file(weight_path),
            )
        )
    return fingerprints


def checksum_first_existing(paths: Iterable[Path]) -> Optional[str]:
    """Return the checksum of the first existing candidate file."""
    for path in paths:
        if path.is_file():
            return sha256_file(path)
    return None


def fingerprint_local_tokenizer(tokenizer: str | Path) -> Optional[str]:
    """Hash every local tokenizer asset that can affect request tokenization."""
    tokenizer_path = Path(tokenizer)
    if tokenizer_path.is_file():
        return sha256_file(tokenizer_path)
    if not tokenizer_path.is_dir():
        return None
    files = [
        tokenizer_path / name for name in LOCAL_TOKENIZER_FILES if (tokenizer_path / name).is_file()
    ]
    if not files:
        return None
    return sha256_json(
        [
            {
                "path": path.relative_to(tokenizer_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    )


def fingerprint_local_model_metadata(model: str | Path) -> Optional[str]:
    """Hash local runtime metadata beyond the primary architecture config."""
    model_path = Path(model)
    if not model_path.is_dir():
        return None
    files = [
        model_path / name for name in LOCAL_MODEL_METADATA_FILES if (model_path / name).is_file()
    ]
    if not files:
        return None
    return sha256_json(
        [
            {
                "path": path.relative_to(model_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    )


def _run_readonly(command: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment_fingerprint() -> EnvironmentFingerprint:
    """Collect best-effort package, CUDA, GPU, CPU, and memory identity."""
    packages: dict[str, Optional[str]] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    gpu_rows: list[dict[str, Any]] = []
    driver_version: Optional[str] = None
    query = _run_readonly(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if query:
        for line in query.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                driver_version = parts[4]
                gpu_rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "uuid": parts[2],
                        "memory_total_mb": float(parts[3]),
                    }
                )

    cuda_version: Optional[str] = None
    try:
        import torch

        cuda_version = torch.version.cuda
    except (ImportError, OSError):
        pass

    memory_bytes: Optional[int] = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        memory_bytes = int(page_size * page_count)
    except (OSError, ValueError):
        pass

    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        cuda_version=cuda_version,
        driver_version=driver_version,
        gpu=gpu_rows,
        cpu=platform.processor() or None,
        memory_bytes=memory_bytes,
    )


def git_state(repository: str | Path) -> tuple[Optional[str], bool, Optional[str]]:
    """Return commit, dirty state, and a human-readable status snapshot."""
    root = Path(repository)
    commit = _run_readonly(["git", "rev-parse", "HEAD"], root)
    status = _run_readonly(["git", "status", "--porcelain=v1"], root)
    return commit, bool(status), status


def source_tree_sha256(repository: str | Path) -> Optional[str]:
    """Hash tracked and untracked, non-ignored source content for strict resume.

    A commit plus a dirty boolean cannot distinguish two different dirty trees.
    Git supplies the execution-relevant path set while file bytes, executable
    modes, symlink targets, and tracked deletions provide the content identity.
    """
    root = Path(repository).resolve()
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    digest = hashlib.sha256(b"slotune-source-tree-v1\0")
    for raw_path in sorted(path for path in completed.stdout.split(b"\0") if path):
        path = root / os.fsdecode(raw_path)
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        try:
            metadata = path.lstat()
        except OSError:
            digest.update(b"\0missing\0")
            continue
        digest.update(stat.S_IMODE(metadata.st_mode).to_bytes(4, "big"))
        if path.is_symlink():
            digest.update(b"\0symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"\0file\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"\0other\0")
    return digest.hexdigest()


def build_manifest(
    *,
    experiment_id: str,
    model: str,
    trace_path: str | Path,
    workload: dict[str, Any],
    slo: dict[str, Any],
    constraints: dict[str, Any],
    gpu_config: dict[str, Any],
    telemetry: dict[str, Any],
    study: dict[str, Any],
    vllm_args: dict[str, Any],
    search_space: dict[str, Any],
    seed: int,
    repository: str | Path,
    model_revision: Optional[str] = None,
    tokenizer: Optional[str] = None,
    holdout_trace_path: Optional[str | Path] = None,
) -> ExperimentSpec:
    """Create a complete immutable experiment manifest."""
    model_path = Path(model)
    tokenizer_path = Path(tokenizer or model)
    model_weight_files = fingerprint_local_model_weights(model_path)
    model_weights_sha256 = (
        sha256_json([item.model_dump(mode="json") for item in model_weight_files])
        if model_weight_files
        else None
    )
    commit, dirty, _ = git_state(repository)
    source_hash = source_tree_sha256(repository)
    identity_config = {
        "model": model,
        "model_revision": model_revision,
        "tokenizer": tokenizer,
        "workload": workload,
        "slo": slo,
        "constraints": constraints,
        "gpu": gpu_config,
        "telemetry": telemetry,
        "study": study,
        "vllm_args": vllm_args,
        "search_space": search_space,
    }
    return ExperimentSpec(
        experiment_id=experiment_id,
        model=model,
        model_revision=model_revision,
        tokenizer=tokenizer,
        model_config_sha256=checksum_first_existing([model_path / "config.json"]),
        model_metadata_sha256=fingerprint_local_model_metadata(model_path),
        tokenizer_sha256=fingerprint_local_tokenizer(tokenizer_path),
        model_weight_files=model_weight_files,
        model_weights_sha256=model_weights_sha256,
        trace_sha256=sha256_file(trace_path),
        holdout_trace_sha256=(
            sha256_file(holdout_trace_path) if holdout_trace_path is not None else None
        ),
        workload=workload,
        slo=slo,
        constraints=constraints,
        gpu_config=gpu_config,
        telemetry=telemetry,
        study=study,
        vllm_args=vllm_args,
        search_space=search_space,
        search_space_sha256=sha256_json(search_space),
        experiment_config_sha256=sha256_json(identity_config),
        seed=seed,
        environment=collect_environment_fingerprint(),
        source_commit=commit,
        source_tree_sha256=source_hash,
        dirty_worktree=dirty,
    )


RESUME_IDENTITY_FIELDS = (
    "model",
    "model_revision",
    "tokenizer",
    "model_config_sha256",
    "model_metadata_sha256",
    "tokenizer_sha256",
    "model_weight_files",
    "model_weights_sha256",
    "trace_sha256",
    "holdout_trace_sha256",
    "workload",
    "slo",
    "constraints",
    "gpu_config",
    "telemetry",
    "study",
    "vllm_args",
    "search_space_sha256",
    "experiment_config_sha256",
    "seed",
    "environment",
    "source_commit",
    "source_tree_sha256",
    "dirty_worktree",
    "artifact_schema_version",
)


def validate_resume_manifest(existing: ExperimentSpec, requested: ExperimentSpec) -> None:
    """Reject silent appends to an incompatible experiment."""
    mismatches = [
        field
        for field in RESUME_IDENTITY_FIELDS
        if getattr(existing, field) != getattr(requested, field)
    ]
    if mismatches:
        raise ValueError(
            "Cannot resume an incompatible experiment; mismatched fields: " + ", ".join(mismatches)
        )
