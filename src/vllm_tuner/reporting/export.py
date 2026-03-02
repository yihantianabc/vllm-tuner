"""Export and import configurations."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


def export_config(
    config: Dict[str, Any],
    output_path: Path,
    format: str = "yaml",
) -> Path:
    """Export configuration to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    elif format == "yaml":
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Configuration exported to: {output_path}")
    return output_path


def import_config(
    config_path: Path,
) -> Dict[str, Any]:
    """Import configuration from file."""
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif suffix in [".yaml", ".yml"]:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def export_best_config(
    best_result: Dict[str, Any],
    output_path: Path,
    format: str = "yaml",
) -> Path:
    """Export best configuration from tuning study."""
    params = best_result.get("parameters", {})
    metrics = best_result.get("metrics", {})

    config = {
        "vllm_params": params,
        "performance_metrics": {
            "throughput_requests_per_sec": metrics.get("throughput_requests_per_sec"),
            "avg_latency_ms": metrics.get("avg_latency_ms"),
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "p99_latency_ms": metrics.get("p99_latency_ms"),
            "throughput_tokens_per_sec": metrics.get("throughput_tokens_per_sec"),
        },
        "trial_info": {
            "trial_number": best_result.get("trial_number"),
            "state": best_result.get("state"),
        },
    }

    return export_config(config, output_path, format)


def export_study_summary(
    study_summary: Dict[str, Any],
    trials_data: list[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Export complete study summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(study_summary, f, indent=2)
    exported["summary"] = summary_path

    trials_path = output_dir / "trials.json"
    with open(trials_path, "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=2)
    exported["trials"] = trials_path

    best = study_summary.get("best_trial", {})
    if best:
        best_yaml = output_dir / "best_config.yaml"
        export_best_config(best, best_yaml, "yaml")
        exported["best_yaml"] = best_yaml

        best_json = output_dir / "best_config.json"
        export_best_config(best, best_json, "json")
        exported["best_json"] = best_json

    logger.info(f"Study summary exported to: {output_dir}")
    return exported
