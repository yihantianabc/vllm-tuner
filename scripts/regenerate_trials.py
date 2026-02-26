#!/usr/bin/env python3
"""Regenerate trials.json and reports from Optuna database."""

import json
import sys
from pathlib import Path
from datetime import datetime

import optuna


def regenerate_trials_from_db(
    study_name: str,
    db_path: str,
    output_dir: Path,
):
    """Load all trials from Optuna database and regenerate reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{db_path}"

    print(f"Loading study '{study_name}' from database...")
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"Total trials in study: {len(study.trials)}")

    all_trials = []

    for trial in study.trials:
        trial_data = {
            "trial_number": trial.number,
            "value": trial.values if len(study.directions) > 1 else trial.value,
            "parameters": trial.params,
            "metrics": trial.user_attrs.get("metrics", {}),
            "state": str(trial.state),
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat()
            if trial.datetime_complete
            else None,
        }
        all_trials.append(trial_data)

    trials_path = configs_dir / "trials.json"
    with open(trials_path, "w") as f:
        json.dump(all_trials, f, indent=2)

    print(f"Exported {len(all_trials)} trials to {trials_path}")

    if len(study.directions) == 1:
        best_trial = study.best_trial
        best_data = {
            "trial_number": best_trial.number,
            "value": best_trial.values if len(study.directions) > 1 else best_trial.value,
            "parameters": best_trial.params,
            "metrics": best_trial.user_attrs.get("metrics", {}),
            "state": str(best_trial.state),
            "datetime_start": best_trial.datetime_start.isoformat()
            if best_trial.datetime_start
            else None,
            "datetime_complete": best_trial.datetime_complete.isoformat()
            if best_trial.datetime_complete
            else None,
        }
    else:
        best_trials = study.best_trials
        best_data = {
            "trial_number": best_trials[0].number,
            "value": best_trials[0].values,
            "parameters": best_trials[0].params,
            "metrics": best_trials[0].user_attrs.get("metrics", {}),
            "state": str(best_trials[0].state),
            "datetime_start": best_trials[0].datetime_start.isoformat()
            if best_trials[0].datetime_start
            else None,
            "datetime_complete": best_trials[0].datetime_complete.isoformat()
            if best_trials[0].datetime_complete
            else None,
        }

    summary_data = {
        "study_name": study_name,
        "num_trials": len(study.trials),
        "study_storage": storage_url,
        "best_trial": best_data,
        "top_n": sorted(
            all_trials, key=lambda x: (x["state"] != "TrialState.COMPLETE", x["trial_number"])
        )[:10],
    }

    summary_path = configs_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Updated summary: {summary_path}")

    print("\nNow regenerate the report with:")
    print(f"  vllm-tuner report --study-name {study_name} --format html")

    return trials_path, summary_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python regenerate_trials.py <study_name> <study_directory>")
        print(
            "Example: python regenerate_trials.py 4c_16g_4090_qwen3_06b studies/4c_16g_4090_qwen3_06b"
        )
        sys.exit(1)

    study_name = sys.argv[1]
    study_dir = Path(sys.argv[2])
    db_path = study_dir / "optuna.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    regenerate_trials_from_db(study_name, str(db_path), study_dir)
