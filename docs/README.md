# vLLM-Tuner Documentation

This documentation covers everything from quick start to advanced features for both users and developers.

## 📚 Documentation Guide

### For Users

#### Getting Started
- [User Guide](user-guide/index.md) - Complete user guide with quick start
- [Installation](user-guide/installation.md) - How to install vLLM-Tuner
- [Configuration](user-guide/configuration.md) - Configuration options explained
- [CLI Commands](user-guide/cli-commands.md) - Command-line reference

#### Learn by Examples
- [Examples](user-guide/examples/) - Ready-to-use configuration examples
  - [Simple Tune](user-guide/examples/simple_tune.yaml) - Basic tuning study
  - [Multi-GPU Tuning](user-guide/examples/multi_gpu_tune.yaml) - Scale across multiple GPUs
  - [Latency Optimization](user-guide/examples/latency_optimized.yaml) - Minimize latency
  - [Custom Workload](user-guide/examples/custom-workload.md) - Use your own dataset

#### Understanding Reports
- [HTML Reports](user-guide/reports/html-reports.md) - Interactive report features
- [Metrics Explained](user-guide/reports/metrics-explained.md) - What each metric means
- [Baseline Comparison](user-guide/reports/baseline-comparison.md) - Comparing with defaults

### For Developers

#### Development
- [Developer Guide](developer-guide/index.md) - Development overview
- [Setup](developer-guide/setup.md) - Development environment setup
- [Testing](developer-guide/testing.md) - Testing guide
- [Contributing](developer-guide/contributing.md) - Contribution guidelines
- [API Reference](developer-guide/api-reference.md) - API documentation

#### Code Quality
- [Code Style](../AGENTS.md) - Coding standards for project

### Architecture

- [Architecture Overview](architecture/index.md) - System design and components
- [Tuning Engine](architecture/tuning-engine.md) - How optimization works

### Troubleshooting

- [Common Issues](troubleshooting/common-issues.md) - Common problems and solutions
- [OOM Errors](troubleshooting/oom-errors.md) - Handling out-of-memory errors

## 📝 Documentation Feedback

If you find errors, have suggestions, or want to improve the documentation:
- Open an issue on GitHub describing the problem or suggestion
- Submit a pull request with your improvements
- See [Contributing](developer-guide/contributing.md) for details on how to contribute

---

**Last Updated:** 2026-02-26
**vLLM-Tuner Version:** 0.1.0