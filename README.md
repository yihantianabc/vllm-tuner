# vLLM Auto-Tuner

An intelligent auto-tuner for vLLM that automatically monitors GPU metrics, uses Bayesian optimization to tune parameters (`batch_size`, `max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization`), and strives to maximize throughput while minimizing latency, respecting user-provided constraints.

## Features

- **Intelligent Profiling**: Monitor GPU memory, utilization, and vLLM metrics automatically
- **Adaptive Parameter Search**: Bayesian optimization (Optuna) with multi-objective support (throughput, latency, memory)
- **vLLM-Aware Integration**: Parse vLLM logs for KV cache utilization, preemption tracking, and guidance
- **Multi-GPU Support**: Handle data-parallel and model-parallel (tensor/pipeline) configurations
- **User-Friendly Configuration**: Simple YAML configs to specify objectives and constraints
- **Rich Reporting**: Plotly interactive HTML reports with trial progression, Pareto front, and GPU telemetry
- **Extensibility**: Custom workloads and plugins for specific deployment scenarios

## Installation

```bash
# Install dependencies using uv
uv venv --seed --python 3.10
source .venv/bin/activate
uv pip install -e .

# Install vLLM
uv pip install vllm --torch-backend=auto
```

**Requirements:**
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)

## Configuration

Configuration is done via YAML file under `config/default.yaml`. Key settings:

### Multi-Objective Weights (must sum to 100)
```yaml
objectives:
  throughput: 60  # Weight for throughput maximization
  latency: 30     # Weight for latency minimization
  memory: 10      # Weight for memory efficiency
```

### Search Space
```yaml
search_space:
  batch_size: [1, 256]  # Range or override defaults
  gpu_memory_utilization: [0.6, 0.99]
  tensor_parallel_size: [1, 2, 4]
```

### Workload
```yaml
workload:
  dataset_name: "tatsu-lab/alpaca"  # HF dataset
  sample_size: 100                 # Number of prompts
  concurrent_requests: 10          # Concurrent clients
```

## Quick Start

### Basic Tuning
```bash
# Run tuning study
vllm-tuner tune --config config/default.yaml --study-name my_study
```

### Multi-GPU Tuning
```bash
vllm-tuner tune --config examples/multi_gpu_tune.yaml --study-name llama2_7b_tune
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `vllm-tuner tune` | Run a tuning study |
| `vllm-tuner report` | Generate reports from completed study |
| `vllm-tuner export` | Export best configuration |
| `vllm-tuner list-studies` | List all available studies |

```bash
# Run tuning study
vllm-tuner tune --config <config.yaml> --study-name <name>

# Generate report (html, json, markdown)
vllm-tuner report --study-name <name> --format html

# Export best config
vllm-tuner export --study-name <name> --format yaml
```

## Documentation

For detailed information, see the [comprehensive documentation](docs/README.md):

### User Guide
- [Quick Start Guide](docs/user-guide/index.md) - Getting started with vLLM-Tuner
- [Installation](docs/user-guide/installation.md) - Detailed installation instructions
- [Configuration](docs/user-guide/configuration.md) - All configuration options
- [CLI Commands](docs/user-guide/cli-commands.md) - Complete command reference
- [Examples](docs/user-guide/examples/) - Ready-to-use configuration examples
  - [Simple Tune](docs/user-guide/examples/simple_tune.yaml) - Basic tuning study
  - [Latency Optimized](docs/user-guide/examples/latency_optimized.yaml) - Minimize latency
  - [Multi-GPU](docs/user-guide/examples/multi_gpu_tune.yaml) - Scale across GPUs

### Reports & Metrics
- [HTML Reports](docs/user-guide/reports/html-reports.md) - Interactive report features
- [Metrics Explained](docs/user-guide/reports/metrics-explained.md) - What each metric means
- [Baseline Comparison](docs/user-guide/reports/baseline-comparison.md) - Comparing with defaults

### Developer Guide
- [Developer Setup](docs/developer-guide/setup.md) - Development environment setup
- [Testing Guide](docs/developer-guide/testing.md) - Running and writing tests
- [Contributing](docs/developer-guide/contributing.md) - How to contribute
- [Code Style](AGENTS.md) - Coding standards and guidelines

### Architecture
- [Architecture Overview](docs/architecture/index.md) - System design and components
- [Baseline Integration](docs/architecture/baseline-integration.md) - Baseline system architecture
- [Tuning Engine](docs/architecture/tuning-engine.md) - How optimization works

### Troubleshooting
- [Common Issues](docs/troubleshooting/common-issues.md) - Common problems and solutions
- [OOM Errors](docs/troubleshooting/oom-errors.md) - Handling out-of-memory errors

## Optimization Objectives

### Throughput Maximization
- Measures requests/second or tokens/second
- Priority for batch processing scenarios

### Latency Minimization
- Measures average request completion time (ms)
- Priority for interactive/real-time applications

### Memory Efficiency
- Measures GPU memory utilization
- Helps fit larger models or more concurrent requests

## Best Practices

1. **Start Simple**: Use `docs/user-guide/examples/simple_tune.yaml` for quick iterations
2. **Set Realistic Constraints**: Avoid overly strict latency/memory targets
3. **Check GPU Availability**: Verify CUDA_VISIBLE_DEVICES before multi-GPU tuning
4. **Review Reports**: HTML reports show Pareto front for trade-off analysis
5. **Export Best Config**: Reuse optimal parameters across runs

## Output Structure

Studies are saved to `studies/<study_name>/`:
```text
studies/my_study/
├── optuna.db                 # SQLite study database
├── baseline/                 # Baseline metrics (if enabled)
│   ├── baseline_metrics.json
│   └── baseline_config.yaml
├── logs/                     # vLLM server logs
├── configs/                  # Summary & best configs
│   ├── summary.json
│   ├── trials.json
│   ├── best_config.yaml
│   └── best_config.json
└── reports/
    └── report.html           # Interactive Plotly report
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint code
ruff check src/ tests/
black src/

# Type check
mypy src/
```

## Troubleshooting

### vLLM Server Not Starting
- Check CUDA_VISIBLE_DEVICES
- Verify GPU availability with `nvidia-smi`
- Review vLLM logs in `studies/<study_name>/logs/`

### OOM Errors
- Reduce `gpu_memory_utilization` upper bound
- Lower `max_num_seqs`
- Check batch_size upper bound

### Slow Optimization
- Reduce `sample_size` (fewer prompts)
- Lower `min_trials`
- Decrease `timeout_minutes`

## Architecture

```text
src/
├── cli/              # Typer CLI commands
├── config/           # Pydantic models & validation
├── tuner/            # Optuna study manager & optimizer
├── vllm/             # vLLM launcher & telemetry parser
├── profiling/        # GPS collectors & vLLM metrics
├── benchmarks/       # Workload loader & request generator
├── reporting/        # HTML reports & progress dashboard
└── optimization/     # Search space definition
```

## License

Apache-2.0

## Acknowledgments

- [Optuna](https://optuna.org/) for Bayesian optimization
- [vLLM](https://github.com/vllm-project/vllm) for high-performance serving
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) for workloads