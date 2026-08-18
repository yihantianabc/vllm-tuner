"""Contract tests for SLOTune documentation and checked-in YAML examples."""

import subprocess
import sys
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


def test_readme_preserves_attribution_and_current_v5_evidence_boundaries() -> None:
    """Project identity, ownership, current evidence, and negative results stay prominent."""

    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    compact_readme = " ".join(readme.split())
    attribution = "forked from"
    ownership = "What is original work"
    required = (
        "# SLOTune: artifact-backed vLLM long-context optimization",
        attribution,
        ownership,
        "Javad Anaraki",
        "Qwen2.5-7B-Instruct",
        "RTX 5090",
        "KV Capacity Planner",
        "Maximum absolute observed KV-block/concurrency error was **0.0446%**",
        "decode-tail-1024",
        "42.8% on the target trace and 42.9% on held-out",
        "-0.027%/-0.014%",
        "FP8 incompatibility",
        "M4 selection stayed `production-default`",
        "original M5 formal artifact",
        "upstream vLLM APC",
        "upstream vLLM Chunked Prefill",
        "does **not** claim to have implemented APC",
        "docs/results/longctx-v5.md",
        "docs/CAREER_MATERIALS.md",
        "docs/results/qwen25-3b-34a25a2.md",
        "scripts/generate_longctx_v5_figures.py",
    )
    for text in required:
        assert text in compact_readme
    assert compact_readme.index(ownership) < compact_readme.index(attribution)


def test_reproduction_guide_separates_audit_offline_and_fresh_gpu_paths() -> None:
    """Reference roots stay read-only and FP8 incompatibility has no retry path."""

    reproduction = (REPOSITORY / "REPRODUCTION.md").read_text(encoding="utf-8")
    required = (
        "read-only audit",
        "offline regeneration",
        "fresh GPU reproduction",
        "./scripts/setup_data_disk_reproduction.sh",
        "validate_m5_decode_tail_artifacts",
        "scripts/generate_longctx_v5_figures.py",
        "--config experiments/long_context/v5/m1-capacity-formal.yaml",
        "--config experiments/long_context/v5/m3-apc-formal.yaml",
        "--config experiments/long_context/v5/m4-chunked-formal.yaml",
        "--config experiments/long_context/v5/m5-decode-tail-formal.yaml",
        "longctx-v5-m5-decode-tail-formal-repro-001",
        "vllm_tuner.longctx.m5_decode_tail_engineering",
        "M2 is inspect-only on this stack",
        "Do not launch `m2-fp8-formal.yaml`",
        "Never point a fresh run or derived output at a reference root",
        "The v2 command must report `GPU runs executed: 0`",
    )
    for text in required:
        assert text in reproduction
    assert "run_longctx_m2_fp8.sh --config" not in reproduction


def test_current_v5_result_and_career_materials_preserve_claim_boundaries() -> None:
    """The public result and resume wording retain ownership and negative-result limits."""

    result = (REPOSITORY / "docs/results/longctx-v5.md").read_text(encoding="utf-8")
    career = (REPOSITORY / "docs/CAREER_MATERIALS.md").read_text(encoding="utf-8")
    for text in (
        "M2 — FP8 incompatibility is a result, not a missing chart",
        "M4 — Chunked Prefill calibration kept production default",
        "Original formal negative and versioned engineering analysis",
        "42.834%",
        "42.899%",
        "2.625 MiB",
        "APC and Chunked Prefill were used and analyzed; neither was authored",
    ):
        assert text in result
    for text in (
        "推荐简历成稿",
        "独立设计并实现 KV Capacity Planner",
        "使用 upstream 原生 Chunked Prefill",
        "FP8 KV 不兼容",
        "M4 的 production-default selection",
        "不推荐的表述",
    ):
        assert text in career
    for name in (
        "longctx-v5-capacity-planner.png",
        "longctx-v5-apc.png",
        "longctx-v5-decode-tail.png",
    ):
        assert (REPOSITORY / "docs/results" / name).is_file()


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
    if sys.platform != "win32":
        for script in (
            "data_disk_reproduction_env.sh",
            "setup_data_disk_reproduction.sh",
            "run_data_disk_reproduction.sh",
            "run_reproduction_command.sh",
        ):
            subprocess.run(["bash", "-n", str(REPOSITORY / "scripts" / script)], check=True)


def test_ci_python_matrices_match_the_declared_package_floor() -> None:
    """CI never schedules a Python version rejected by project metadata."""

    for relative_path in (".github/workflows/cli.yml", ".github/workflows/release.yml"):
        workflow = (REPOSITORY / relative_path).read_text(encoding="utf-8")
        assert '"3.9"' not in workflow
        for version in ("3.10", "3.11", "3.12"):
            assert f'"{version}"' in workflow


def test_project_plan_audit_tracks_completed_evidence_and_deferrals() -> None:
    """The M0-M6 audit must prove core completion without hiding deferred work."""

    audit = (REPOSITORY / "docs/PLAN_AUDIT.md").read_text(encoding="utf-8")
    compact_audit = " ".join(audit.split())
    for milestone in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        assert f"{milestone}:" in audit
    required = (
        "Complete plan-section coverage",
        "§0–2 identity, goals, value",
        "§24 official references",
        "Implementation and test evidence",
        "Completed formal evidence",
        "Core M0–M5 complete; optional M6 deferred explicitly",
        "Definition-of-Done audit",
        "Core Definition of Done complete with negative performance outcome",
        "ad36ee8",
        "143 entries in total, including 96 trial anchors",
        "89 COMPLETE, seven constraint-INFEASIBLE",
        "Chat yields only a tested",
        "M6 prefix-caching",
    )
    for text in required:
        assert text in compact_audit


def test_demo_script_has_valid_shell_syntax_and_explicit_output_contract() -> None:
    """The short demo is runnable and never chooses an implicit artifact path."""

    script = REPOSITORY / "scripts/run_demo.sh"
    if sys.platform != "win32":
        subprocess.run(["bash", "-n", str(script)], check=True)
    contents = script.read_text(encoding="utf-8")
    assert "[[ $# -lt 1 || $# -gt 2 ]]" in contents
    assert 'OUTPUT_DIR="$1"' in contents
    assert 'FORMAL_ROOT="${2:-}"' in contents
    assert '--output-dir "${OUTPUT_DIR}"' in contents
    assert "report/report.html" in contents
    assert "report/capacity-curve.png" in contents
    assert "report/scheduler-negative-results.md" in contents
    assert "no GPU experiment was launched" in contents


def test_formal_result_snapshot_records_negative_results_and_claim_boundaries() -> None:
    """The checked-in result index must retain exact provenance and negative outcomes."""

    snapshot = (REPOSITORY / "docs/results/qwen25-3b-34a25a2.md").read_text(encoding="utf-8")
    compact_snapshot = " ".join(snapshot.split())
    required = (
        "34a25a2e10951bfab1c2a86b4c60aff5bef785df",
        "ad36ee8e0e15a6d0502a35f9e794b056b9522a82",
        "8ea95533232bf6b0d45b75513ec4c799f3ab42595fb66abd5e9893142fbfae7a",
        "7d704beea1890d14f7a411d677b867cdc8a06584a5040dbde2793f6723c8e191",
        "7df0229c115ec0ce41cbc3c72624b13597b2a33d8f93a762242dbe723ca498b7",
        "ade1eaa13a4f78c49c498404c100f2e5458c6a194b1d378ccda283d415a04361",
        "c78bdb8d57c5deef51053f41d4e50d8d48f9fe0ee9b5d069220d3a562f138c8b",
        "b9da5621b4f075b387a1e2be93968294367249a205992b0c7cffe6acb5895e2f",
        "3d382db39c7279b27752137567cc7779c510fd6419f6f995aa29608285b5e1e3",
        "89e5d6f9e505f72aec5594323c4fc6f3e35ed5c0fa7d7927d4d1b1ff63b1c6b9",
        "aac7760903a8c609756fbead137a694065dd8c1193081e97f6884653c0200b84",
        "d92f7fc83a04e57aaa424ef9da1e92ecd40f33fe00247f45dde75583f3af0c57",
        "f13d91219e7d18f15c5fcb5f4d2f00f138ac1c39ba124717849277c52322c1fd",
        "89 COMPLETE and seven constraint-INFEASIBLE",
        "48,000 measured requests per workload",
        "8.029529 req/s",
        "8.456300 req/s",
        "4.254534 req/s",
        "4.331800 req/s",
        "Empirical scheduled req/s",
        "0.944593",
        "30.226983",
        "1.018862",
        "32.603574",
        "legacy `measured_offered_requests_per_sec` is a target-rate alias",
        "at least 15% more goodput",
        "20% lower p99 TTFT",
        "capacity lower bound of at least 27.883 req/s",
        "two of three repeats are INFEASIBLE",
        "exactly 0% goodput gain",
        "CPU simulator output",
        "M6 prefix-caching/APC experiments are deferred P1 work",
        "chat-capacity.png",
        "rag-capacity.png",
        "chat-pareto.png",
        "rag-pareto.png",
        "experiment-integrity.json",
        "lineage.json",
        "experiment-audit.json",
        "summary.compact-v1.json",
        "scheduler-negative-results.json",
        "report/scheduler-negative-results.md",
        "ArtifactStore.attest_experiment_artifacts",
        "ArtifactStore.validate_experiment_integrity",
        "vllm-tuner attest",
        "--reseal",
    )
    for text in required:
        assert text in compact_snapshot


def test_completed_docs_contain_no_stale_formal_pending_placeholders() -> None:
    """Completed evidence pages cannot regress to the pre-measurement placeholder state."""

    paths = (
        REPOSITORY / "README.md",
        REPOSITORY / "REPRODUCTION.md",
        REPOSITORY / "docs/FORMAL_EXPERIMENTS.md",
        REPOSITORY / "docs/PLAN_AUDIT.md",
        REPOSITORY / "docs/DEVELOPMENT_LOG.md",
        REPOSITORY / "docs/results/qwen25-3b-34a25a2.md",
    )
    stale_phrases = (
        "Pending final commit",
        "Pending an immutable artifact",
        "Status: NOT YET RECORDED",
        "Formal evidence pending",
        "does **not** satisfy the plan's experimental Definition of Done",
        "once executed after the tool revision",
        "must be executed only after its implementation revision is committed",
        "提交后再对 formal roots 执行",
    )
    for path in paths:
        contents = path.read_text(encoding="utf-8")
        for phrase in stale_phrases:
            assert phrase not in contents, (path, phrase)
