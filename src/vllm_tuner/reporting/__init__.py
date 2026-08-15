"""Reporting module for vLLM tuner."""

from .html import generate_html_report
from .export import export_study_summary, export_best_config
from .dashboard import ProgressDashboard
from .plots import (
    capacity_curve,
    save_figure,
    search_comparison,
    search_trajectory,
    summarize_capacity_rows,
    telemetry_timeline,
)
from .report import generate_report

__all__ = [
    "generate_html_report",
    "export_study_summary",
    "export_best_config",
    "ProgressDashboard",
    "capacity_curve",
    "generate_report",
    "save_figure",
    "search_comparison",
    "search_trajectory",
    "summarize_capacity_rows",
    "telemetry_timeline",
]
