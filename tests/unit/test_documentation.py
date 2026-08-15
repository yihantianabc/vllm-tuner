"""Contract tests for SLOTune documentation and checked-in YAML examples."""

import subprocess
from pathlib import Path

import yaml

from vllm_tuner.config.validation import load_yaml_config

REPOSITORY = Path(__file__).resolve().parents[2]


def documented_yaml_files() -> list[Path]:
    """Return runtime and user-guide YAML files that promise current-schema validity."""

    paths = sorted((REPOSITORY / "config").glob("*.yaml"))
    paths.extend(sorted((REPOSITORY / "docs/user-guide/examples").glob("*.yaml")))
    return paths


def test_all_documented_yaml_examples_load_with_current_schema() -> None:
    """No checked-in example contains stale or unknown configuration fields."""

    paths = documented_yaml_files()
    assert paths
    for path in paths:
        config = load_yaml_config(path)
        assert config.gpu.count == 1, path
        assert config.search_space.tensor_parallel_size == 1, path
        assert config.search_space.pipeline_parallel_size == 1, path


def test_documented_yaml_does_not_reintroduce_legacy_objective_or_batch_size() -> None:
    """Weighted objectives and ineffective batch_size stay out of executable examples."""

    for path in documented_yaml_files():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "objectives" not in raw, path
        assert "batch_size" not in raw.get("search_space", {}), path


def test_formal_3b_configs_have_equal_budget_repeats_and_holdout() -> None:
    """Chat and RAG formal protocols carry the required evidence controls."""

    profiles = {}
    for name in ("formal_3b_chat.yaml", "formal_3b_rag.yaml"):
        path = REPOSITORY / "config" / name
        config = load_yaml_config(path)
        assert "3B" in config.model
        assert config.study.methods == ["default", "random", "tpe"]
        assert config.study.trial_budget > 0
        assert config.study.repeat_count == 3
        assert config.study.holdout_enabled
        assert config.workload.sample_size >= 500
        assert config.workload.fixed_output_tokens == 128
        assert config.workload.ignore_eos
        assert config.workload.benchmark_backend == "sse"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["workload"]["capacity_request_rates"] == [1, 2, 4, 8, 16, 32]
        assert raw["workload"]["capacity_repeats"] == 3
        profiles[config.workload.name] = config
    assert profiles.keys() == {"chat", "rag"}


def test_readme_preserves_fork_attribution_and_evidence_boundaries() -> None:
    """Project identity and smoke/formal boundaries remain prominent."""

    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    attribution = "Forked from jranaraki/vllm-tuner."
    focus = (
        "My work focuses on benchmark correctness, SLO-aware optimization, cross-layer "
        "observability, reproducibility, and scheduling experiments."
    )
    required = (
        "# SLOTune",
        attribution,
        focus,
        "Javad Anaraki",
        "My Contributions",
        "| Area | SLOTune contribution | Code | Tests | Artifact or evidence | Commit |",
        "src/vllm_tuner/benchmarks/sse_client.py",
        "tests/unit/test_benchmark_sse_client.py",
        "docs/METHODOLOGY.md#artifact-acceptance",
        "Pending final commit",
        "Qwen3-0.6B",
        "smoke test only",
        "config/formal_3b_chat.yaml",
        "config/formal_3b_rag.yaml",
        "qwen25-3b-preflight-20260815-a",
        "Pending an immutable artifact",
        "Limitations and future work",
    )
    for text in required:
        assert text in readme
    assert readme.index(attribution) < readme.index(focus)


def test_reproduction_guide_separates_legacy_smoke_preflight_and_formal() -> None:
    """Reproduction commands cannot turn historical or tiny runs into formal evidence."""

    reproduction = (REPOSITORY / "REPRODUCTION.md").read_text(encoding="utf-8")
    required = (
        "legacy bring-up artifact",
        "historical boundary",
        "reproduction_gpu_20260815_a",
        "smoke-20260815-b",
        "qwen25-3b-preflight-20260815-a",
        "two-request default run",
        "./scripts/setup_data_disk_reproduction.sh",
        "./scripts/run_data_disk_reproduction.sh slotune_smoke_001",
        "./scripts/run_reproduction_command.sh tune",
        "--config config/formal_3b_chat.yaml",
        "--config config/formal_3b_rag.yaml",
        "--trace /root/autodl-tmp/traces/chat-search.jsonl",
        "/root/autodl-tmp/slotune-results/<study-name>",
        "Current real-results register",
        "Status: NOT YET RECORDED",
        "No claim",
    )
    for text in required:
        assert text in reproduction


def test_reproduction_environment_is_locked_and_inherited_by_formal_commands() -> None:
    """The GPU overlay stays core-compatible and formal children inherit data-disk paths."""

    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (REPOSITORY / "requirements-reproduction.txt").read_text(encoding="utf-8")
    assert '"transformers>=4.56.0,<5"' in pyproject
    assert '"numpy>=1.25,<2.3"' in pyproject
    assert '"idna>=3.18"' in pyproject
    for requirement in (
        "vllm==0.16.0",
        "transformers==4.57.6",
        "numpy==2.2.6",
        "idna==3.18",
    ):
        assert requirement in requirements

    setup = (REPOSITORY / "scripts/setup_data_disk_reproduction.sh").read_text(encoding="utf-8")
    assert setup.index("uv sync --extra dev --frozen --inexact") < setup.index("uv pip install")
    assert setup.index("uv pip install") < setup.index("uv pip check")

    environment = (REPOSITORY / "scripts/data_disk_reproduction_env.sh").read_text(encoding="utf-8")
    for variable in (
        "UV_CACHE_DIR",
        "TMPDIR",
        "HF_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "VLLM_CACHE_ROOT",
        "FLASHINFER_WORKSPACE_BASE",
    ):
        assert f"export {variable}=" in environment

    wrapper = (REPOSITORY / "scripts/run_reproduction_command.sh").read_text(encoding="utf-8")
    assert 'source "${SCRIPT_DIR}/data_disk_reproduction_env.sh"' in wrapper
    assert 'CLI="${SLOTUNE_REPO_DIR}/.venv/bin/vllm-tuner"' in wrapper
    assert 'exec "${CLI}" "$@"' in wrapper
    for script in (
        "data_disk_reproduction_env.sh",
        "setup_data_disk_reproduction.sh",
        "run_data_disk_reproduction.sh",
        "run_reproduction_command.sh",
    ):
        subprocess.run(["bash", "-n", str(REPOSITORY / "scripts" / script)], check=True)


def test_project_plan_audit_tracks_implementation_and_evidence_separately() -> None:
    """The M0-M6 audit must expose formal evidence gaps instead of implying completion."""

    audit = (REPOSITORY / "docs/PLAN_AUDIT.md").read_text(encoding="utf-8")
    for milestone in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        assert f"{milestone}:" in audit
    required = (
        "Complete plan-section coverage",
        "§0–2 identity, goals, value",
        "§24 official references",
        "Implementation and test evidence",
        "Formal evidence pending",
        "3B pipeline preflight—not the formal chat/RAG experiment",
        "Definition-of-Done audit",
        "**Not complete**; 3B preflight is not formal",
        "does **not** satisfy the plan's experimental Definition of Done",
    )
    for text in required:
        assert text in audit


def test_demo_script_has_valid_shell_syntax_and_explicit_output_contract() -> None:
    """The short demo is runnable and never chooses an implicit artifact path."""

    script = REPOSITORY / "scripts/run_demo.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    contents = script.read_text(encoding="utf-8")
    assert "[[ $# -ne 1 ]]" in contents
    assert '--output-dir "$1"' in contents
