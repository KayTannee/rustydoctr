# Rusty doctr

Bounded Rust/CUDA word OCR with a Python streaming interface and reproducible docTR comparisons.
Rust code is reserved for `src/`; supporting tools live in `scripts/`, `pybaseline/`, and `pytests/`.

- [Benchmark/profile a machine and save its parameters](BENCHMARK.md)
- [Build the Python library and stream rendered pages](PYTHON.md)
- [Measured throughput and Python boundary results](RESULTS.md)
- [Opt-in dense-text refinement and its benchmark](DENSE_REFINEMENT.md)
- [Dense-tile parameter tuning: results and reproduction](DENSE_TUNING.md)
- [Missing ordinary words: revised priority and visual audit](MISSING_WORDS.md)
- [Opt-in native thin-character recovery and measured results](THIN_RECOVERY.md)
- [Remaining omissions: stage audit and one-hop context experiment](REMAINING_OMISSIONS.md)
- [Opt-in page orientation and fractional deskew](PAGE_ORIENTATION.md)
- [Head-to-head docTR/Rust rotation comparison](ROTATION_COMPARISON.md)
- [Experimental line-guided crops and error audit](LINE_ORIENTATION.md)
- [Merged-word contact sheet and next experiments](MERGE_AUDIT.md)
- [Segmentation and rotation accuracy experiments](QUALITY_FINDINGS.md)

See [the benchmark guide](pybaseline/README.md). Generated report: `pybaseline/results/report.html`.

The first native implementation is available: [Rust vertical slice](RUST_SLICE.md).

For the bounded CPU/GPU pipeline, automatic batch calibration and sustained
Python comparisons, see [the throughput guide](THROUGHPUT.md).

Public labelled dataset choices and published-score comparison protocol: [PUBLIC_BENCHMARKS.md](PUBLIC_BENCHMARKS.md).
