# vLLM-Tuner Documentation

Welcome to the vLLM-Tuner documentation hub. This documentation covers everything from quick start to advanced features for both users and developers.

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
- [Baseline Integration](architecture/baseline-integration.md) - Baseline system architecture
- [Tuning Engine](architecture/tuning-engine.md) - How optimization works

### Troubleshooting

- [Common Issues](troubleshooting/common-issues.md) - Common problems and solutions
- [OOM Errors](troubleshooting/oom-errors.md) - Handling out-of-memory errors

## 🚀 Quick Links

### For New Users
- [Quick Start](user-guide/index.md#quick-start)
- [Installation](user-guide/installation.md)
- [First Tuning Study](user-guide/examples/README.md)

### For Existing Users
- [Advanced Configuration](user-guide/configuration.md)
- [Examples](user-guide/examples/)
- [Understanding Reports](user-guide/reports/)

### For Developers
- [Development Setup](developer-guide/setup.md)
- [Running Tests](developer-guide/testing.md)
- [Contributing](developer-guide/contributing.md)

## 📖 External Resources

- [vLLM Documentation](https://docs.vllm.ai/) - Official vLLM documentation
- [Optuna Documentation](https://optuna.readthedocs.io/) - Bayesian optimization by Preferred Networks
- [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation using Python type annotations
- [Plotly Documentation](https://plotly.com/python/) - Python graphing library for HTML reports

## 🤝 Contributing to Documentation

This documentation is open source and we welcome contributions! See the [Contributing Guide](developer-guide/contributing.md) for details on how to improve the documentation.

## 🔍 Finding Information

### By Topic

- **Configuration**: [Configuration](user-guide/configuration.md), [Examples](user-guide/examples/)
- **Running Studies**: [CLI Commands](user-guide/cli-commands.md), [Examples](user-guide/examples/)
- **Understanding Results**: [Reports](user-guide/reports/)
- **Development**: [Developer Guide](developer-guide/)
- **Troubleshooting**: [Troubleshooting](troubleshooting/)
- **Architecture**: [Architecture](architecture/)

### By User Type

- **End Users**: Start with [User Guide](user-guide/index.md)
- **Developers**: Start with [Developer Guide](developer-guide/index.md)
- **System Integrators**: See [Architecture](architecture/) and [API Reference](developer-guide/api-reference.md)

### Common Tasks

| Task | Documentation |
|------|---------------|
| Install vLLM-Tuner | [Installation](user-guide/installation.md) |
| Run your first study | [User Guide → Examples](user-guide/examples/) |
| Configure optimization | [Configuration](user-guide/configuration.md) |
| Generate reports | [Reports](user-guide/reports/) |
| Setup development env | [Developer Setup](developer-guide/setup.md) |
| Write tests | [Testing Guide](developer-guide/testing.md) |
| Contribute code | [Contributing](developer-guide/contributing.md) |

## 📝 Documentation Feedback

If you find errors, have suggestions, or want to improve the documentation:
- Open an issue on GitHub describing the problem or suggestion
- Submit a pull request with your improvements
- See [Contributing](developer-guide/contributing.md) for details on how to contribute

---

**Last Updated:** 2026-02-26
**vLLM-Tuner Version:** 0.1.0