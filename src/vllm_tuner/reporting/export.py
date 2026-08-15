"""Export and import configurations."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)


def _canonical_vllm_argument(name: str) -> str:
    return name.strip().lstrip("-").replace("-", "_")


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
    best_result: Mapping[str, Any],
    output_path: Path,
    format: str = "yaml",
    *,
    experiment_id: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
    validation: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Export a legacy or current best record without aliasing absent metrics."""
    params_value = best_result.get("parameters")
    if not isinstance(params_value, Mapping) or not params_value:
        raise ValueError("best result does not contain non-empty parameters")
    params = dict(params_value)
    nested_metrics = best_result.get("metrics")
    metrics = nested_metrics if isinstance(nested_metrics, Mapping) else best_result
    repeat_value = best_result.get("repeat_metrics")
    repeat_metrics = dict(repeat_value) if isinstance(repeat_value, Mapping) else {}
    holdout_value = best_result.get("holdout_metrics")
    holdout_metrics = dict(holdout_value) if isinstance(holdout_value, Mapping) else {}
    search_value = best_result.get("search_observation")
    search_observation = dict(search_value) if isinstance(search_value, Mapping) else {}
    manifest_data = dict(manifest or {})
    model = manifest_data.get("model")
    if manifest is not None and (not isinstance(model, str) or not model):
        raise ValueError("manifest.model is required for a replayable export")
    base_args_value = manifest_data.get("vllm_args", {})
    if base_args_value is None:
        base_args_value = {}
    if not isinstance(base_args_value, Mapping):
        raise ValueError("manifest.vllm_args must be an object")
    base_args = dict(base_args_value)
    if any(not isinstance(key, str) for key in (*base_args.keys(), *params.keys())):
        raise ValueError("vLLM argument names must be strings")
    replay_args = dict(base_args)
    for parameter_name, parameter_value in params.items():
        canonical = _canonical_vllm_argument(parameter_name)
        aliases = [name for name in replay_args if _canonical_vllm_argument(name) == canonical]
        conflicting = [name for name in aliases if replay_args[name] != parameter_value]
        if conflicting:
            raise ValueError(
                "validated parameters conflict with manifest.vllm_args: "
                + ", ".join(sorted(conflicting))
            )
        for alias in aliases:
            del replay_args[alias]
        replay_args[parameter_name] = parameter_value

    config: Dict[str, Any] = {
        # Retain the established key for consumers of older exports while the
        # optional replay block carries the complete server configuration.
        "vllm_params": params,
        "performance_metrics": {
            "goodput_requests_per_sec": metrics.get("goodput_requests_per_sec"),
            "offered_requests_per_sec": metrics.get("offered_requests_per_sec"),
            "achieved_requests_per_sec": metrics.get("achieved_requests_per_sec"),
            "p99_ttft_ms": metrics.get("p99_ttft_ms"),
            "p99_tpot_ms": metrics.get("p99_tpot_ms"),
            "p99_e2e_ms": metrics.get("p99_e2e_ms"),
            "throughput_requests_per_sec": metrics.get("throughput_requests_per_sec"),
            "avg_latency_ms": metrics.get("avg_latency_ms"),
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "p99_latency_ms": metrics.get("p99_latency_ms"),
            "throughput_tokens_per_sec": metrics.get("throughput_tokens_per_sec"),
        },
        "metric_provenance": best_result.get("metric_provenance"),
        "repeat_metrics": repeat_metrics,
        "holdout_metrics": holdout_metrics,
        "search_observation": search_observation,
        "candidate_info": {
            "candidate": best_result.get("candidate"),
            "method": best_result.get("method"),
            "status": best_result.get("status", best_result.get("state")),
            "validated": best_result.get("validated"),
        },
    }
    if manifest is not None:
        config.update(
            {
                "model": model,
                "model_revision": manifest_data.get("model_revision"),
                "tokenizer": manifest_data.get("tokenizer"),
                "vllm_args": replay_args,
                "base_vllm_args": base_args,
                "validated_best_parameters": params,
            }
        )
    if experiment_id is not None:
        config["source_experiment"] = experiment_id
    if validation is not None:
        config["validation"] = dict(validation)

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
