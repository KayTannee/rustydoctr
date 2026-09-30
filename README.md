# Rusty doctr

Bounded Rust/CUDA word OCR with a Python streaming interface and reproducible docTR comparisons.
Rust code lives in `src/`; supporting tools live in `scripts/`, `pybaseline/`, and `pytests/`.

- [Install the Python package on another Windows NVIDIA PC](docs/guides/PYTHON_DEPLOYMENT.md)
- [Build the Python library and stream rendered pages](docs/guides/PYTHON.md)
- [Benchmark/profile a machine and save its parameters](docs/guides/BENCHMARK.md)
- [Measured throughput and Python boundary results](docs/benchmarks/RESULTS.md)
- [Full documentation index](docs/README.md): guides, benchmarks, quality experiments, architecture and third-party notices.

See the [Python baseline guide](pybaseline/README.md) for docTR setup.
Generated reports remain in `pybaseline/results/`.
