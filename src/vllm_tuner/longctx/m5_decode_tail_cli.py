"""Standalone command-line entry point for long-context v5 M5."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .m5_decode_tail_config import load_longctx_m5_decode_tail_config
from .m5_decode_tail_runner import LongContextM5DecodeTailRunner, load_m5_decode_tail_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect long-context v5 M5 evidence")
    parser.add_argument(
        "-c",
        "--config",
        default="experiments/long_context/v5/m5-decode-tail-smoke.yaml",
        help="Strict M5 smoke or formal YAML",
    )
    parser.add_argument(
        "--experiment-id",
        default="longctx-v5-m5-decode-tail-smoke-001",
        help="M5 artifact directory name",
    )
    parser.add_argument("--resume", action="store_true", help="Resume checksum-valid pairs")
    parser.add_argument("--status", action="store_true", help="Read status without a model")
    parser.add_argument(
        "--artifact-root",
        default="/root/autodl-tmp/longctx-v5-artifacts",
        help="Artifact root used with --status",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_longctx_m5_decode_tail_config(args.config)
    runner = LongContextM5DecodeTailRunner(
        config,
        args.experiment_id,
        repository=Path(__file__).resolve().parents[3],
        resume=args.resume,
    )
    summary = asyncio.run(runner.run())
    execution = summary.get("execution", {})
    acceptance = summary.get("acceptance", {})
    analysis = summary.get("analysis", {})
    decision = analysis.get("decision", {}) if isinstance(analysis, Mapping) else {}
    execution_passed = isinstance(execution, Mapping) and execution.get("passed") is True
    acceptance_passed = isinstance(acceptance, Mapping) and acceptance.get("passed") is True
    print(f"Execution: {'PASS' if execution_passed else 'FAIL'}")
    if isinstance(execution, Mapping):
        print(
            "M5 jobs: "
            f"{execution.get('completed_jobs', 'unavailable')}/"
            f"{execution.get('planned_jobs', 'unavailable')} completed; "
            f"{execution.get('failed_jobs', 'unavailable')} failed"
        )
    label = "M5 formal acceptance" if config.evidence_role == "formal" else "M5 smoke"
    print(f"{label}: {'PASS' if acceptance_passed else 'FAIL'}")
    if isinstance(decision, Mapping):
        print(f"Decision: {decision.get('wording')}")
    print(f"Artifacts: {runner.store.root}")
    print("M6 was not started.")
    return 0 if acceptance_passed else 1


def _status(args: argparse.Namespace) -> int:
    status = load_m5_decode_tail_status(args.artifact_root, args.experiment_id)
    for label, key in (
        ("State", "state"),
        ("PID", "pid"),
        ("GPU", "gpu"),
        ("Progress", "completed_jobs"),
        ("Planned", "planned_jobs"),
        ("Current trial", "current_trial"),
        ("Log", "log"),
        ("Result", "result"),
        ("ETA", "eta"),
        ("Resume", "resume"),
        ("Sealed", "sealed"),
        ("Acceptance", "acceptance"),
    ):
        value = status.get(key)
        print(
            f"{label}: {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run M5 and return a shell-compatible exit status."""
    args = _parser().parse_args(argv)
    try:
        return _status(args) if args.status else _run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
