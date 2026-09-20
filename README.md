# Rusty doctr

Bounded Rust/CUDA word OCR with a Python streaming interface and reproducible docTR comparisons.
Rust code is reserved for `src/`; supporting tools live in `scripts/`, `pybaseline/`, and `pytests/`.

- [Benchmark/profile a machine and save its parameters](BENCHMARK.md)
- [Build the Python library and stream rendered pages](PYTHON.md)
- [Measured throughput and Python boundary results](RESULTS.md)
- [Opt-in dense-text refinement and its benchmark](DENSE_REFINEMENT.md)
- [Opt-in page orientation and fractional deskew](PAGE_ORIENTATION.md)
- [Head-to-head docTR/Rust rotation comparison](ROTATION_COMPARISON.md)
- [Segmentation and rotation accuracy experiments](QUALITY_FINDINGS.md)

See [the benchmark guide](pybaseline/README.md). Generated report: `pybaseline/results/report.html`.

The first native implementation is available: [Rust vertical slice](RUST_SLICE.md).

For the bounded CPU/GPU pipeline, automatic batch calibration and sustained
Python comparisons, see [the throughput guide](THROUGHPUT.md).
