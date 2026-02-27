# vLLM-Tuner

<p align="center">
    <img src="docs/logo.png" alt="vllm-tuner" style="width:50%; height:auto;">
</p>

An intelligent tuner for vLLM that automatically monitors GPU metrics, uses Bayesian optimization to tune parameters (`batch_size`, `max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization`) to maximize throughput while minimizing latency and balancing memory, respecting user-provided constraints.

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
# Create and activate uv environment
uv venv --seed --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install vLLM
uv pip install vllm --torch-backend=auto
```

## Configuration

Configuration is done via YAML file under `config/default.yaml`, and here are the key settings:

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

## Run

### Basic Tuning
```bash
# Run tuning study
vllm-tuner tune --config config/default.yaml --study-name my_study
```

### Multi-GPU Tuning
```bash
vllm-tuner tune --config examples/multi_gpu_tune.yaml --study-name llama2_7b_tune
```

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

## Documentation

For detailed information, see the [comprehensive documentation](docs/README.md).

## Acknowledgments

- [Optuna](https://optuna.org/) for Bayesian optimization
- [vLLM](https://github.com/vllm-project/vllm) for high-performance serving
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) for workloads