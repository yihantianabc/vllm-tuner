"""Generate interactive HTML reports with Plotly."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import yaml

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("plotly is required. Install with: pip install plotly")

import jinja2

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate interactive HTML reports with Plotly charts."""

    def __init__(
        self, study_name: str, output_dir: Path, baseline_data: Optional[Dict[str, Any]] = None
    ):
        self.study_name = study_name
        self.output_dir = output_dir
        self.baseline_data = baseline_data
        output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        study_summary: Dict[str, Any],
        trials_data: list[Dict[str, Any]],
    ) -> Path:
        """Generate complete HTML report."""
        logger.info(f"Generating HTML report for study: {self.study_name}")

        charts = self._create_charts(study_summary, trials_data)

        output_path = self.output_dir / "report.html"

        html_content = self._render_html(
            study_summary,
            trials_data,
            charts,
        )

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _load_baseline_data(self, study_dir: Path) -> Optional[Dict[str, Any]]:
        """Load baseline metrics from baseline directory."""
        baseline_dir = study_dir / "baseline"

        try:
            baseline_file = baseline_dir / "baseline_metrics.json"
            if baseline_file.exists():
                with open(baseline_file) as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    logger.info(f"Loaded baseline metrics from {baseline_file}")
                    return metrics

            baseline_yaml = baseline_dir / "baseline_config.yaml"
            if baseline_yaml.exists():
                with open(baseline_yaml) as f:
                    data = yaml.safe_load(f)
                    metrics = data.get("baseline_metrics", {})
                    logger.info(f"Loaded baseline metrics from {baseline_yaml}")
                    return metrics
        except Exception as e:
            logger.warning(f"Failed to load baseline data: {e}")

        return None

    def _create_charts(
        self,
        study_summary: Dict[str, Any],
        trials_data: list[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Create Plotly charts for the report."""
        charts = {}

        if not trials_data:
            logger.warning("No trial data available for charts")
            return charts

        # Filter out failed trials for charts
        # Trials are considered failed if:
        # - state doesn't contain "COMPLETE" (handles both "COMPLETE" and "TrialState.COMPLETE")
        # - oom_detected is True
        # - throughput is 0 or inf
        # - latency is inf
        successful_trials = []
        failed_count = 0

        for t in trials_data:
            state = t.get("state", "UNKNOWN")
            metrics = t.get("metrics", {})

            # Check if trial is successful
            # State can be "COMPLETE", "TrialState.COMPLETE", or similar
            is_complete = "COMPLETE" in state.upper()
            has_oom = metrics.get("oom_detected", False)
            has_error = "error" in metrics

            throughput = metrics.get("throughput_requests_per_sec", 0)
            latency = metrics.get("avg_latency_ms", 0)

            has_valid_throughput = throughput > 0 and throughput != float("inf")
            has_valid_latency = latency != float("inf")

            if (
                is_complete
                and not has_oom
                and not has_error
                and has_valid_throughput
                and has_valid_latency
            ):
                successful_trials.append(t)
            else:
                failed_count += 1

        if not successful_trials:
            logger.warning("No successful trials available for charts")
            return charts

        if failed_count > 0:
            logger.info(
                f"Filtered out {failed_count} failed trials from charts, using {len(successful_trials)} successful trials"
            )

        trial_numbers = [t.get("trial_number", i) for i, t in enumerate(successful_trials)]
        throughputs = [
            t.get("metrics", {}).get("throughput_requests_per_sec", 0) for t in successful_trials
        ]
        latencies = [t.get("metrics", {}).get("avg_latency_ms", 0) for t in successful_trials]
        memory_utils = [
            t.get("metrics", {}).get("aggregate_memory_utilization", 0) for t in successful_trials
        ]

        charts["throughput_progression"] = self._create_throughput_chart(
            trial_numbers,
            throughputs,
        )

        charts["latency_distribution"] = self._create_latency_chart(
            trial_numbers,
            latencies,
        )

        charts["pareto_front"] = self._create_pareto_chart(
            throughputs,
            latencies,
        )

        charts["gpu_memory"] = self._create_memory_chart(
            trial_numbers,
            memory_utils,
        )

        charts["combined"] = self._create_combined_chart(
            trial_numbers,
            throughputs,
            latencies,
            memory_utils,
        )

        return charts

    def _create_throughput_chart(
        self,
        trial_numbers: list[int],
        throughputs: list[float],
    ) -> str:
        """Create throughput progression chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=throughputs,
                mode="lines+markers",
                name="Throughput",
                line=dict(color="#2ecc71"),
            )
        )

        fig.update_layout(
            title="Throughput Over Trials",
            xaxis_title="Trial Number",
            yaxis_title="Throughput (requests/sec)",
            hovermode="x unified",
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _create_latency_chart(
        self,
        trial_numbers: list[int],
        latencies: list[float],
    ) -> str:
        """Create latency distribution chart."""
        valid_latencies = [l for l in latencies if l < float("inf")]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=latencies,
                mode="lines+markers",
                name="Avg Latency",
                line=dict(color="#f39c12"),
            )
        )

        fig.update_layout(
            title="Average Latency Over Trials",
            xaxis_title="Trial Number",
            yaxis_title="Latency (ms)",
            hovermode="x unified",
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _create_pareto_chart(
        self,
        throughputs: list[float],
        latencies: list[float],
    ) -> str:
        """Create Pareto front scatter plot."""
        valid_data = [(t, l) for t, l in zip(throughputs, latencies) if t > 0 and l < float("inf")]

        if not valid_data:
            return ""

        valid_throughputs = [d[0] for d in valid_data]
        valid_latencies = [d[1] for d in valid_data]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=valid_throughputs,
                y=valid_latencies,
                mode="markers",
                name="Trials",
                marker=dict(
                    size=10,
                    color="#3498db",
                    line=dict(width=1, color="#2980b9"),
                ),
            )
        )

        fig.update_layout(
            title="Pareto Front: Throughput vs. Latency",
            xaxis_title="Throughput (requests/sec)",
            yaxis_title="Average Latency (ms)",
            hovermode="closest",
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _create_memory_chart(
        self,
        trial_numbers: list[int],
        memory_utils: list[float],
    ) -> str:
        """Create GPU memory utilization chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=memory_utils,
                mode="lines+markers",
                name="Memory Utilization",
                line=dict(color="#9b59b6"),
            )
        )

        fig.update_layout(
            title="GPU Memory Utilization Over Trials",
            xaxis_title="Trial Number",
            yaxis_title="Memory Utilization",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _create_combined_chart(
        self,
        trial_numbers: list[int],
        throughputs: list[float],
        latencies: list[float],
        memory_utils: list[float],
    ) -> str:
        """Create combined multi-panel chart."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Throughput",
                "Latency",
                "Memory Utilization",
                "Pareto Front",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=throughputs,
                mode="lines+markers",
                name="Throughput",
                line=dict(color="#2ecc71"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=latencies,
                mode="lines+markers",
                name="Latency",
                line=dict(color="#f39c12"),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=memory_utils,
                mode="lines+markers",
                name="Memory",
                line=dict(color="#9b59b6"),
            ),
            row=2,
            col=1,
        )

        valid_data = [(t, l) for t, l in zip(throughputs, latencies) if t > 0 and l < float("inf")]

        if valid_data:
            valid_throughputs = [d[0] for d in valid_data]
            valid_latencies = [d[1] for d in valid_data]

            fig.add_trace(
                go.Scatter(
                    x=valid_throughputs,
                    y=valid_latencies,
                    mode="markers",
                    name="Pareto",
                    marker=dict(
                        size=8,
                        color="#3498db",
                        opacity=0.7,
                    ),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            height=800,
            title_text=f"Trial Results: {self.study_name}",
            showlegend=False,
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _render_html(
        self,
        study_summary: Dict[str, Any],
        trials_data: list[Dict[str, Any]],
        charts: Dict[str, str],
    ) -> str:
        """Render HTML from template."""
        best_trial = study_summary.get("best_trial", {})
        best_metrics = best_trial.get("metrics", {})
        best_params = best_trial.get("parameters", {})

        baseline_metrics = self.baseline_data or {}
        baseline_available = bool(baseline_metrics)

        baseline_throughput = baseline_metrics.get("throughput_requests_per_sec", 0)
        baseline_latency = baseline_metrics.get("avg_latency_ms", 0)
        baseline_p95 = baseline_metrics.get("p95_latency_ms", 0)
        baseline_memory = baseline_metrics.get("average_memory_mb", 0)
        baseline_p99 = baseline_metrics.get("p99_latency_ms", 0)

        best_throughput = best_metrics.get("throughput_requests_per_sec", 0)
        best_latency = best_metrics.get("avg_latency_ms", 0)
        best_p95 = best_metrics.get("p95_latency_ms", 0)
        best_memory = best_metrics.get("average_memory_mb", 0)
        best_p99 = best_metrics.get("p99_latency_ms", 0)

        throughput_improvement = (
            ((best_throughput - baseline_throughput) / baseline_throughput * 100)
            if baseline_throughput > 0
            else 0
        )
        latency_improvement = (
            ((baseline_latency - best_latency) / baseline_latency * 100)
            if baseline_latency > 0
            else 0
        )
        p95_improvement = (
            ((baseline_p95 - best_p95) / baseline_p95 * 100) if baseline_p95 > 0 else 0
        )
        memory_delta = (
            ((best_memory - baseline_memory) / baseline_memory * 100) if baseline_memory > 0 else 0
        )
        p99_improvement = (
            ((baseline_p99 - best_p99) / baseline_p99 * 100) if baseline_p99 > 0 else 0
        )

        abs_latency_improvement = abs(latency_improvement)
        abs_p95_improvement = abs(p95_improvement)
        abs_p99_improvement = abs(p99_improvement)
        abs_memory_delta = abs(memory_delta)

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM Tuner Report - {{ study_name }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .header { background: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 28px; font-weight: bold; color: #2ecc71; }
        .metric-label { color: #7f8c8d; font-size: 14px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .params-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .params-table th, .params-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .params-table th { background: #f8f9fa; }
        .timestamp { color: #95a5a6; font-size: 14px; }
        .positive { color: #2ecc71; font-weight: bold; }
        .negative { color: #e74c3c; font-weight: bold; }
        .neutral { color: #7f8c8d; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>vLLM Tuner Report</h1>
        <p class="timestamp">Generated: {{ generation_time }}</p>
        <h2>Study: {{ study_name }}</h2>
        
        <h3>Best Configuration Metrics</h3>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(best_throughput) }}</div>
                <div class="metric-label">Throughput (req/s)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(best_latency) }}</div>
                <div class="metric-label">Avg Latency (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ num_trials }}</div>
                <div class="metric-label">Total Trials</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ trial_number }}</div>
                <div class="metric-label">Best Trial #</div>
            </div>
        </div>

        {% if baseline_available %}
        <h3>Baseline vs Best Trial Comparison</h3>
        <table class="params-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Best Trial</th>
                    <th>Improvement</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Throughput (req/s)</td>
                    <td>{{ "%.2f"|format(baseline_throughput) }}</td>
                    <td>{{ "%.2f"|format(best_throughput) }}</td>
                    <td class="{{ 'positive' if throughput_improvement > 0 else ('negative' if throughput_improvement < 0 else 'neutral') }}">
                        {{ ("+" if throughput_improvement > 0 else "") }}{{ "%.1f"|format(throughput_improvement) }}%
                    </td>
                </tr>
                <tr>
                    <td>Avg Latency (ms)</td>
                    <td>{{ "%.2f"|format(baseline_latency) }}</td>
                    <td>{{ "%.2f"|format(best_latency) }}</td>
                    <td class="{{ 'positive' if latency_improvement > 0 else ('negative' if latency_improvement < 0 else 'neutral') }}">
                        {{ ("+" if latency_improvement > 0 else ("-" if latency_improvement < 0 else "")) }}{{ "%.1f"|format(abs_latency_improvement) }}%
                    </td>
                </tr>
                <tr>
                    <td>P95 Latency (ms)</td>
                    <td>{{ "%.2f"|format(baseline_p95) }}</td>
                    <td>{{ "%.2f"|format(best_p95) }}</td>
                    <td class="{{ 'positive' if p95_improvement > 0 else ('negative' if p95_improvement < 0 else 'neutral') }}">
                        {{ ("+" if p95_improvement > 0 else ("-" if p95_improvement < 0 else "")) }}{{ "%.1f"|format(abs_p95_improvement) }}%
                    </td>
                </tr>
                <tr>
                    <td>P99 Latency (ms)</td>
                    <td>{{ "%.2f"|format(baseline_p99) }}</td>
                    <td>{{ "%.2f"|format(best_p99) }}</td>
                    <td class="{{ 'positive' if p99_improvement > 0 else ('negative' if p99_improvement < 0 else 'neutral') }}">
                        {{ ("+" if p99_improvement > 0 else ("-" if p99_improvement < 0 else "")) }}{{ "%.1f"|format(abs_p99_improvement) }}%
                    </td>
                </tr>
                <tr>
                    <td>Avg Memory (MB)</td>
                    <td>{{ "%.0f"|format(baseline_memory) }}</td>
                    <td>{{ "%.0f"|format(best_memory) }}</td>
                    <td class="{{ 'positive' if memory_delta < 0 else 'negative' if memory_delta > 0 else 'neutral' }}">
                        {{ ("-" if memory_delta < 0 else ("+" if memory_delta > 0 else "")) }}{{ "%.1f"|format(abs_memory_delta) }}%
                    </td>
                </tr>
            </tbody>
        </table>
        {% endif %}

        <h3>Best Parameters</h3>
        <table class="params-table">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>
                {% for param, value in best_params.items() %}
                <tr><td>{{ param }}</td><td>{{ value }}</td></tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <h2>Performance Charts</h2>
    
    <div class="chart-container">
        <h3>Throughput Progression</h3>
        {{ charts.get('throughput_progression', '')|safe }}
    </div>
    
    <div class="chart-container">
        <h3>Latency Distribution</h3>
        {{ charts.get('latency_distribution', '')|safe }}
    </div>
    
    <div class="chart-container">
        <h3>Pareto Front: Throughput vs. Latency</h3>
        {{ charts.get('pareto_front', '')|safe }}
    </div>
    
    <div class="chart-container">
        <h3>GPU Memory Utilization</h3>
        {{ charts.get('gpu_memory', '')|safe }}
    </div>
    
    <div class="chart-container">
        <h3>Combined View</h3>
        {{ charts.get('combined', '')|safe }}
    </div>
</body>
</html>
        """

        template = jinja2.Template(html_template)

        return template.render(
            study_name=self.study_name,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            best_throughput=best_throughput,
            best_latency=best_latency,
            best_p95=best_p95,
            best_p99=best_p99,
            best_memory=best_memory,
            num_trials=study_summary.get("num_trials", 0),
            trial_number=best_trial.get("trial_number", 0),
            best_params=best_params,
            charts=charts,
            baseline_available=baseline_available,
            baseline_throughput=baseline_throughput,
            baseline_latency=baseline_latency,
            baseline_p95=baseline_p95,
            baseline_p99=baseline_p99,
            baseline_memory=baseline_memory,
            throughput_improvement=throughput_improvement,
            latency_improvement=latency_improvement,
            p95_improvement=p95_improvement,
            p99_improvement=p99_improvement,
            memory_delta=memory_delta,
            abs_latency_improvement=abs_latency_improvement,
            abs_p95_improvement=abs_p95_improvement,
            abs_p99_improvement=abs_p99_improvement,
            abs_memory_delta=abs_memory_delta,
        )


def generate_html_report(
    study_name: str,
    output_dir: Path,
    study_summary: Dict[str, Any],
    trials_data: list[Dict[str, Any]],
    baseline_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate HTML report with given data."""
    generator = HTMLReportGenerator(study_name, output_dir, baseline_data)
    return generator.generate_report(study_summary, trials_data)
