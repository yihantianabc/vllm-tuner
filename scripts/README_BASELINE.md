# Baseline Metrics Generator

Generate baseline performance metrics for vLLM models using default parameters.

## Overview

This script generates baseline throughput, latency, and GPU memory utilization metrics for vLLM models. It uses:

- **vLLM default parameters**: `gpu_memory_utilization=0.9`, `max_num_seqs=128`, `max_num_batched_tokens=2048`
- **Alpaca dataset**: Loads 1000 prompts from the HC datasets
- **5 warmup requests**: Stabilize the server before benchmarking
- **10 concurrent requests**: Simulates realistic load
- **Continuous GPU monitoring**: Tracks memory, utilization, and temperature throughout benchmark

## Installation

Ensure the required packages are installed:

```bash
pip install datasets transformers yaml pynvml httpx
```

## Usage

### Basic Example

```bash
python scripts/generate_baseline.py --model "Qwen/Qwen2.5-0.5B" --gpu-ids 0
```

### With Custom Parameters

```bash
python scripts/generate_baseline.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --gpu-ids 0,1 \
  --num-requests 1000 \
  --concurrency 10 \
  --warmup 5 \
  --max-tokens 256 \
  --output-dir baselines/llama2_7b
```

### Multi-GPU

```bash
python scripts/generate_baseline.py \
  --model "meta-llama/Llama-2-70b-hf" \
  --gpu-ids 0,1,2,3 \
  --output-dir baselines/llama2_70b_4gpu
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt2` | Model name or HuggingFace path |
| `--gpu-ids` | `0` | Comma-separated GPU device IDs |
| `--num-requests` | `1000` | Number of benchmark requests |
| `--concurrency` | `10` | Concurrent requests |
| `--warmup` | `5` | Warmup requests |
| `--max-tokens` | `256` | Max output tokens per request |
| `--host` | `127.0.0.1` | vLLM server host |
| `--port` | `8000` | vLLM server port |
| `--dataset` | `tatsu-lab/alpaca` | HuggingFace dataset for prompts |
| `--output-dir` | `baselines/<model>_<timestamp>` | Output directory |

## Output Files

The script generates three output files:

### 1. `baseline_metrics.json` (Full details)
```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "timestamp": "2026-02-24T12:00:00",
  "configuration": {
    "vllm_params": {...},
    "benchmark_params": {...}
  },
  "metrics": {
    "throughput_requests_per_sec": 12.34,
    "throughput_tokens_per_sec": 3158,
    "avg_latency_ms": 810.5,
    "p50_latency_ms": 795.2,
    "p95_latency_ms": 1250.3,
    "p99_latency_ms": 1890.7,
    "avg_ttft_ms": 45.2,
    "peak_memory_mb": 8192,
    "average_memory_mb": 7680,
    "average_gpu_utilization": 0.92,
    "max_temperature_c": 78.5,
    ...
  },
  "gpu_info": {...}
}
```

### 2. `baseline_config.yaml` (Concise format)
```yaml
model: "Qwen/Qwen2.5-0.5B"
timestamp: "2026-02-24T12:00:00"
vllm_params:
  gpu_memory_utilization: 0.9
  max_num_seqs: 128
  max_num_batched_tokens: 2048
  tensor_parallel_size: 1
baseline_metrics:
  throughput_requests_per_sec: 12.34
  throughput_tokens_per_sec: 3158
  avg_latency_ms: 810.5
  p50_latency_ms: 795.2
  p95_latency_ms: 1250.3
  p99_latency_ms: 1890.7
  peak_memory_mb: 8192
  average_memory_mb: 7680
  average_gpu_utilization: 0.92
  max_temperature_c: 78.5
  num_completed: 1000
  duration_seconds: 81.05
```

### 3. `baseline_summary.txt` (Human-readable)
```
================================================================================
BASELINE METRICS FOR QWEN/QWEN2.5-0.5B
================================================================================

Configuration:
  - vLLM Parameters (Defaults):
    ├── gpu_memory_utilization: 0.9
    ├── max_num_seqs: 128
    ├── max_num_batched_tokens: 2048
    └── tensor_parallel_size: 1

  - Benchmark Parameters:
    ├── num_requests: 1000
    ├── concurrency: 10
    ├── warmup_requests: 5
    └── max_tokens: 256

Performance Metrics:
  Throughput:         12.34 requests/sec  (3158 tokens/sec)
  Avg Latency:        810.5 ms
  P50 Latency:        795.2 ms
  P95 Latency:        1250.3 ms
  P99 Latency:        1890.7 ms
  TTFT:               45.2 ms

GPU Metrics:
  Initial Memory:     4096 MB
  Peak Memory:        8192 MB
  Avg Memory:         7680 MB
  Utilization:        90.0%
  Avg GPU Util:       92.0%
  Max Temperature:    78.5 C

Requests:  1000 completed / 1000 total
Duration:  81.05 seconds

Generated: 2026-02-24T12:00:00
================================================================================
```

## Output Directory Structure

```
baselines/
└── <model>_<timestamp>/
    ├── baseline_metrics.json      # Full metrics in JSON
    ├── baseline_config.yaml       # Concise YAML format
    ├── baseline_summary.txt       # Human-readable summary
    └── logs/
        └── vllm_baseline.log      # vLLM server logs
```

## vLLM Default Parameters

The script uses vLLM's default parameters:

| Parameter | Default | Description |
|----------|---------|-------------|
| `gpu_memory_utilization` | 0.9 | Fraction of GPU memory to use (0-1) |
| `max_num_seqs` | 128 | Maximum concurrent sequences |
| `max_num_batched_tokens` | 2048 | Maximum tokens in one batch |
| `tensor_parallel_size` | 1 | Tensor parallel groups (for multi-GPU) |
| `pipeline_parallel_size` | 1 | Pipeline parallel groups |

## Metrics Collected

### Throughput Metrics
- `throughput_requests_per_sec`: Requests completed per second
- `throughput_tokens_per_sec`: Output tokens generated per second

### Latency Metrics
- `avg_latency_ms`: Average request completion time
- `p50_latency_ms`: Median latency (50th percentile)
- `p95_latency_ms`: 95th percentile latency
- `p99_latency_ms`: 99th percentile latency
- `avg_ttft_ms`: Average time to first token

### GPU Memory Metrics
- `initial_memory_mb`: GPU memory at start
- `peak_memory_mb`: Peak GPU memory during benchmark
- `average_memory_mb`: Average GPU memory during benchmark

### GPU Telemetry
- `average_gpu_utilization`: Average GPU utilization (%)
- `max_temperature_c`: Peak GPU temperature

### Performance
- `requests_completed`: Total requests completed
- `duration_seconds`: Total benchmark duration
- `errors`: Number of failed requests

## Failure Handling

The script **aborts** if any request fails. This ensures baseline numbers are reliable.

## Streaming Response Handling

The script uses streaming responses to accurately measure:
- Time to First Token (TTFT)
- Token output rate
- Per-request timings

## Example Output

```
$ python scripts/generate_baseline.py --model "Qwen/Qwen2.5-0.5B"

2026-02-24 12:00:00 - src.generate_baseline - INFO - Starting baseline generation for Qwen/Qwen2.5-0.5B
2026-02-24 12:00:00 - src.generate_baseline - INFO - Loading prompts from tatsu-lab/alpaca...
2026-02-24 12:00:05 - src.generate_baseline - INFO - Loaded 1000 prompts
2026-02-24 12:00:05 - src.generate_baseline - INFO - Starting vLLM server with default parameters...
2026-02-24 12:00:05 - src.generate_baseline - INFO - vLLM command: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B ...
2026-02-24 12:00:05 - src.generate_baseline - INFO - vLLM server started with PID 12345
2026-02-24 12:00:15 - src.generate_baseline - INFO - vLLM server ready
2026-02-24 12:00:15 - src.generate_baseline - INFO - Starting GPU monitoring...
2026-02-24 12:00:15 - src.generate_baseline - INFO - Running warmup with 5 requests...
2026-02-24 12:00:18 - src.generate_baseline - INFO - Warmup completed
2026-02-24 12:00:18 - src.generate_baseline - INFO - Starting main benchmark with 1000 requests, concurrency=10
2026-02-24 12:01:39 - src.generate_baseline - INFO - Benchmark completed: 1000/1000 requests succeeded (0 failed)
2026-02-24 12:01:39 - src.generate_baseline - INFO - GPU monitoring stopped
2026-02-24 12:01:39 - src.generate_baseline - INFO - JSON output: baselines/Qwen_Qwen2.5-0.5B_20260224_120000/baseline_metrics.json
2026-02-24 12:01:39 - src.generate_baseline - INFO - YAML output: baselines/Qwen_Qwen2.5-0.5B_20260224_120000/baseline_config.yaml
2026-02-24 12:01:39 - src.generate_baseline - INFO - Text summary: baselines/Qwen_Qwen2.5-0.5B_20260224_120000/baseline_summary.txt
2026-02-24 12:01:40 - src.generate_baseline - INFO - Stopping vLLM server (PID 12345)...
2026-02-24 12:01:41 - src.generate_baseline - INFO - vLLM server stopped gracefully

================================================================================
BASELINE METRICS SUMMARY
================================================================================

Model: Qwen/Qwen2.5-0.5B
Timestamp: 2026-02-24T12:00:00

...

Output directory: baselines/Qwen_Qwen2.5-0.5B_20260224_120000
```

## Notes

- The script requires a CUDA-capable GPU
- Ensure sufficient GPU memory for the model
- For multi-GPU setups, use `--gpu-ids 0,1,2,3`
- vLLM server logs are saved in `logs/vllm_baseline.log`