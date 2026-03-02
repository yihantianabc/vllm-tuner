"""Live progress dashboard using Rich."""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logger = logging.getLogger(__name__)


class ProgressDashboard:
    """Live progress dashboard for tuning studies."""

    def __init__(self, study_name: str):
        self.study_name = study_name
        self.console = Console()
        self.progress: Optional[Progress] = None
        self.trial_task: Optional[int] = None
        self.total_trials: int = 1
        self.current_trial: int = 0
        self.best_throughput = 0.0
        self.best_latency = float("inf")

    def start(self, total_trials: int = 100) -> None:
        """Start the progress dashboard."""
        self.total_trials = total_trials

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

        self.trial_task = self.progress.add_task(
            f"[bold cyan]Tuning {self.study_name}[/bold cyan]",
            total=total_trials,
        )

        self.progress.start()

        self._print_header()

    def _print_header(self) -> None:
        """Print study header."""
        header = f"""
[bold blue]vLLM Auto-Tuner[/bold blue]
[bold yellow]Study: {self.study_name}[/bold yellow]
        """

        panel = Panel(
            header.strip(),
            title="[bold]Study Info[/bold]",
            border_style="blue",
        )

        self.console.print(panel)

    def update_trial(self, trial_num: int) -> None:
        """Update progress for a trial."""
        self.current_trial = trial_num

        if self.progress and self.trial_task is not None:
            self.progress.update(
                self.trial_task,
                advance=1,
                description=f"[bold cyan]Tuning {self.study_name} - Trial {trial_num}[/bold cyan]",
            )

    def update_best_metrics(
        self,
        throughput: float,
        latency: float,
    ) -> None:
        """Update and display best metrics."""
        self.best_throughput = max(self.best_throughput, throughput)
        self.best_latency = min(self.best_latency, latency)

        self._print_metrics_table()

    def _print_metrics_table(self) -> None:
        """Print current best metrics table."""
        table = Table(title="[bold green]Best Metrics[/bold green]")

        table.add_column("[cyan]Metric[/cyan]", style="cyan")
        table.add_column("[yellow]Value[/yellow]", style="yellow")

        table.add_row(
            "Throughput",
            f"[green]{self.best_throughput:.2f}[/green] req/s",
        )
        table.add_row(
            "Latency",
            f"[green]{self.best_latency:.2f}[/green] ms",
        )

        self.console.print(table)

    def log_trial(
        self,
        trial_num: int,
        params: dict,
        metrics: dict,
    ) -> None:
        """Log trial results."""
        throughput = metrics.get("throughput_requests_per_sec", 0.0)
        latency = metrics.get("avg_latency_ms", 0.0)

        table = Table(show_header=False, box=None)

        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Trial #", str(trial_num))
        table.add_row("Throughput", f"{throughput:.2f} req/s")
        table.add_row("Latency", f"{latency:.2f} ms")

        if throughput > self.best_throughput:
            self.update_best_metrics(throughput, latency)

        self.console.print(table)

    def log_message(self, message: str) -> None:
        """Log a message."""
        self.console.print(f"[dim]{message}[/dim]")

    def log_success(self, message: str) -> None:
        """Log a success message."""
        self.console.print(f"[green]✓[/green] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.console.print(f"[red]✗[/red] [red]{message}[/red]")

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.console.print(f"[yellow]⚠[/yellow] [yellow]{message}[/yellow]")

    def show_summary(self, summary: dict) -> None:
        """Show study summary."""
        table = Table(title="[bold blue]Study Summary[/bold blue]")

        table.add_column("[cyan]Metric[/cyan]", style="cyan")
        table.add_column("[yellow]Value[/yellow]", style="yellow")

        table.add_row(
            "Study Name",
            summary.get("study_name", "unknown"),
        )
        table.add_row(
            "Total Trials",
            str(summary.get("num_trials", 0)),
        )

        best = summary.get("best_trial", {})
        metrics = best.get("metrics", {})

        table.add_row(
            "Best Throughput",
            f"{metrics.get('throughput_requests_per_sec', 0):.2f} req/s",
        )
        table.add_row(
            "Best Latency",
            f"{metrics.get('avg_latency_ms', 0):.2f} ms",
        )

        self.console.print(table)

    def stop(self) -> None:
        """Stop the progress dashboard."""
        if self.progress:
            self.progress.stop()
            self.progress = None

        self.console.print("[bold green]Study completed![/bold green]")


class SimpleDashboard:
    """Simpler dashboard without progress bar."""

    def __init__(self, study_name: str):
        self.study_name = study_name
        self.console = Console()

    def log_trial(self, trial_num: int, metrics: dict) -> None:
        """Log trial results."""
        throughput = metrics.get("throughput_requests_per_sec", 0.0)
        latency = metrics.get("avg_latency_ms", 0.0)

        self.console.print(
            f"[cyan]Trial {trial_num}:[/cyan] "
            f"[green]{throughput:.2f}[/green] req/s, "
            f"[yellow]{latency:.2f}[/yellow] ms"
        )

    def log_message(self, message: str) -> None:
        """Log a message."""
        self.console.print(f"[dim]{message}[/dim]")

    def log_error(self, message: str) -> None:
        """Log an error."""
        self.console.print(f"[red]✗[/red] [red]{message}[/red]")

    def show_summary(self, summary: dict) -> None:
        """Show study summary."""
        self.console.print(f"[bold blue]Study: {self.study_name}[/bold blue]")

        best = summary.get("best_trial", {})
        metrics = best.get("metrics", {})

        self.console.print(
            f"Best throughput: [green]{metrics.get('throughput_requests_per_sec', 0):.2f}[/green] req/s"
        )
        self.console.print(
            f"Best latency: [green]{metrics.get('avg_latency_ms', 0):.2f}[/green] ms"
        )
