"""Configuration validation and YAML parsing."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import TuningConfig, TunerSettings


def load_yaml_config(config_path: str | Path) -> TuningConfig:
    """Load and validate YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    try:
        return TuningConfig(**config_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


def save_yaml_config(config: TuningConfig, output_path: str | Path) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> TuningConfig:
    """Get default tuning configuration with preset weights."""
    return TuningConfig()


def validate_study_name(name: str) -> str:
    """Validate study name is safe for file system."""
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars:
        name = name.replace(char, "_")

    if not name:
        raise ValueError("Study name cannot be empty")

    return name


def create_study_dirs(study_name: str, settings: TunerSettings) -> dict[str, Path]:
    """Create necessary directories for a study."""
    study_dir = Path(settings.study_output_dir) / study_name
    reports_dir = Path(settings.html_output_dir) / study_name

    study_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "study": study_dir,
        "reports": reports_dir,
        "logs": study_dir / "logs",
        "configs": study_dir / "configs",
    }
