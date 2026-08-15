"""Unit tests for the standalone deterministic scheduler ablation script."""

import json

import pytest

from scripts.run_scheduler_ablation import (
    JSON_ARTIFACT_NAME,
    MARKDOWN_ARTIFACT_NAME,
    build_parser,
    builtin_traces,
    load_split_trace_jsonl,
    main,
)
from vllm_tuner.scheduling import DEFAULT_FIXED_BUDGETS


def write_jsonl(path, records):
    """Write a small fixed JSONL fixture."""

    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_help_documents_required_output_and_trace_examples():
    """The script is discoverable without external README changes."""

    parser = build_parser()
    help_text = parser.format_help()

    assert "--output-dir" in help_text
    assert "required" in help_text
    assert "JSONL request schema" in help_text
    assert "--trace-jsonl traces/mixed_scheduler.jsonl" in help_text
    assert tuple(parser.get_default("fixed_budgets")) == DEFAULT_FIXED_BUDGETS
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_builtin_traces_are_distinct_and_deterministic():
    """Calibration and holdout defaults are frozen request sequences."""

    first = builtin_traces()
    repeated = builtin_traces()

    assert first == repeated
    assert first[0] != first[1]
    assert any(request.prompt_tokens >= 2048 for request in first[0])
    assert any(request.prompt_tokens >= 2048 for request in first[1])


def test_combined_jsonl_loader_requires_and_preserves_both_splits(tmp_path):
    """One fixed trace file can carry calibration and unseen requests."""

    trace_path = tmp_path / "trace.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "split": "calibration",
                "request_id": "cal",
                "arrival_time": 0.0,
                "prompt_tokens": 16,
                "output_tokens": 2,
            },
            {
                "split": "held_out",
                "request_id": "hold",
                "arrival_time": 0.1,
                "prompt_tokens": 32,
                "output_tokens": 3,
                "priority": 1,
            },
        ],
    )

    calibration, held_out = load_split_trace_jsonl(trace_path)

    assert [request.request_id for request in calibration] == ["cal"]
    assert [request.request_id for request in held_out] == ["hold"]
    assert held_out[0].priority == 1


def test_main_writes_json_and_markdown_with_negative_conditions(tmp_path):
    """The built-in workflow writes both complete, machine-readable artifacts."""

    output_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "--output-dir",
            str(output_dir),
            "--fixed-budgets",
            "512",
            "1024",
        ]
    )

    assert exit_code == 0
    json_path = output_dir / JSON_ARTIFACT_NAME
    markdown_path = output_dir / MARKDOWN_ARTIFACT_NAME
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["trace_source"] == "builtin_deterministic_v1"
    assert payload["fixed_budgets"] == [512, 1024]
    assert payload["trace_sizes"]["calibration"] > 0
    assert payload["trace_sizes"]["held_out"] > 0
    assert payload["schema_version"] == 2
    assert len(payload["trace_sha256"]["calibration"]) == 64
    assert len(payload["trace_sha256"]["held_out"]) == 64
    assert "source_commit" in payload["provenance"]
    assert "source_tree_sha256" in payload["provenance"]
    assert "environment" in payload["provenance"]
    assert payload["report"]["calibration"]["fixed_baselines"].keys() == {
        "512",
        "1024",
    }
    assert payload["negative_gain_conditions"]
    assert "## Negative or no-benefit conditions" in markdown
    assert "## held_out" in markdown
    assert "Source tree SHA-256" in markdown


def test_main_is_byte_reproducible_for_same_seed_and_builtin_trace(tmp_path):
    """No timestamp or output path leaks into deterministic result artifacts."""

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    common = ["--fixed-budgets", "512", "1024", "--seed", "77"]

    main(["--output-dir", str(first_dir), *common])
    main(["--output-dir", str(second_dir), *common])

    assert (first_dir / JSON_ARTIFACT_NAME).read_bytes() == (
        second_dir / JSON_ARTIFACT_NAME
    ).read_bytes()
    assert (first_dir / MARKDOWN_ARTIFACT_NAME).read_bytes() == (
        second_dir / MARKDOWN_ARTIFACT_NAME
    ).read_bytes()


def test_main_accepts_combined_jsonl_and_rejects_implicit_overwrite(tmp_path):
    """A user trace runs end to end and artifacts are protected by default."""

    trace_path = tmp_path / "combined.jsonl"
    output_dir = tmp_path / "result"
    write_jsonl(
        trace_path,
        [
            {
                "split": "calibration",
                "request_id": "cal-0",
                "arrival_time": 0.0,
                "prompt_tokens": 64,
                "output_tokens": 4,
            },
            {
                "split": "calibration",
                "request_id": "cal-1",
                "arrival_time": 0.01,
                "prompt_tokens": 512,
                "output_tokens": 2,
            },
            {
                "split": "held_out",
                "request_id": "hold-0",
                "arrival_time": 0.0,
                "prompt_tokens": 128,
                "output_tokens": 3,
            },
        ],
    )
    arguments = [
        "--trace-jsonl",
        str(trace_path),
        "--output-dir",
        str(output_dir),
        "--fixed-budgets",
        "512",
        "2048",
    ]

    assert main(arguments) == 0
    payload = json.loads((output_dir / JSON_ARTIFACT_NAME).read_text(encoding="utf-8"))
    assert payload["trace_sizes"] == {"calibration": 2, "held_out": 1}
    with pytest.raises(SystemExit):
        main(arguments)
    assert main([*arguments, "--overwrite"]) == 0


@pytest.mark.parametrize(
    "records, message",
    [
        (
            [
                {
                    "request_id": "missing-split",
                    "arrival_time": 0.0,
                    "prompt_tokens": 1,
                    "output_tokens": 1,
                }
            ],
            "split must be",
        ),
        (
            [
                {
                    "split": "calibration",
                    "request_id": "bad-tokens",
                    "arrival_time": 0.0,
                    "prompt_tokens": "one",
                    "output_tokens": 1,
                },
                {
                    "split": "held_out",
                    "request_id": "hold",
                    "arrival_time": 0.0,
                    "prompt_tokens": 1,
                    "output_tokens": 1,
                },
            ],
            "prompt_tokens must be an integer",
        ),
    ],
)
def test_combined_jsonl_validation_reports_actionable_line_error(tmp_path, records, message):
    """Malformed traces fail before an experiment can produce misleading data."""

    trace_path = tmp_path / "invalid.jsonl"
    write_jsonl(trace_path, records)

    with pytest.raises(ValueError, match=message):
        load_split_trace_jsonl(trace_path)


def test_main_rejects_one_fixed_budget_and_unpaired_trace(tmp_path):
    """CLI validation enforces meaningful baselines and a real holdout."""

    trace_path = tmp_path / "single.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "request_id": "one",
                "arrival_time": 0.0,
                "prompt_tokens": 1,
                "output_tokens": 1,
            }
        ],
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir",
                str(tmp_path / "bad-budget"),
                "--fixed-budgets",
                "512",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir",
                str(tmp_path / "bad-pair"),
                "--calibration-trace-jsonl",
                str(trace_path),
            ]
        )
