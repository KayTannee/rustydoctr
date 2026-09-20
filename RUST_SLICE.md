# Rust word-OCR vertical slice

For the newer bounded, overlapping CPU/GPU pipeline and sustained comparison,
see [THROUGHPUT.md](THROUGHPUT.md). This document describes the original sequential
slice and its older task/results.

The native executable in `src/` loads DB ResNet34 and PARSeq exported from the same
docTR 1.1.0 pretrained weights used by the Python experiments. CUDA inference uses
ONNX Runtime 1.23.2 through `ort`. Preprocessing, straight DB box postprocessing,
crop extraction, PARSeq token decoding and word JSON are implemented in Rust.
The executable does not invoke Python. The current launcher finds NVIDIA and ORT
DLLs in the existing Python environment; a standalone distribution would package
those native runtime dependencies separately.

## Run

VS Code: **Terminal → Run Task → Rust: upright throughput (GPU-Z)**.
Choose five or ten minutes. The task builds first, then repeatedly processes the
900-word upright control page. Outputs are timestamped under
`pybaseline/results/manual_rust`. This is a word-only in-memory benchmark, not the
previous Python feeder/writer/rotation benchmark; compare the same-scope reference
runs for speed claims.

```powershell
# Already set up on this workstation. For a fresh environment:
./pybaseline/setup.ps1
./scripts/setup_rust.ps1

# Individual run; output folder must not already exist.
./scripts/run_rust.ps1 --image testdata/generated/a4_control.png --size 1024 --reco-batch 128 --seconds 30 --output pybaseline/results/my_rust_run
```

Model initialization and CUDA algorithm selection can take about a minute on the
first inference in each process. Timing starts after a full warmup. Image decoding,
model loading, warmup and JSON writing are excluded from reported throughput.
The models remain loaded across repeated passes. `--profile` writes ORT traces;
use separate diagnostic runs because profiling changes timings. `--dump` saves
first-page tensors for numerical comparisons. GPU registration failure is fatal;
`--cpu` is an explicit diagnostic opt-in, never an automatic benchmark fallback.

## Scope and known differences

- Upright pages and axis-aligned word boxes only. No page/crop rotation, tiling,
  retries, line/block reconstruction, PDF rendering, or streaming workers yet.
- Pages are processed sequentially; recognition crops are batched within a page.
  The initial default recognition batch is 128. Cross-page batching comes later.
- Detector sides must be multiples of 32; the current wrapper uses square inputs
  and the same symmetric-padding coordinate convention as docTR.
- Crops wider than aspect ratio 8 are rejected with an explicit error. docTR's
  wide-crop splitting/remapping remains to be ported.
- Native antialiased resizing is numerically close, not guaranteed bit-identical
  to Torchvision's uint8 implementation. Saved tensor comparisons quantify this.
- DB's upright rectangle expansion has been reimplemented analytically; 20 oracle
  fixtures from docTR test borders, merging, score thresholds and coordinate mapping.
- PARSeq's ONNX export executes full-length decoding; native PyTorch can stop
  autoregressive decoding early. Exported-model parity and PyTorch parity are
  evaluated separately. ORT may run shape/index operations on CPU even with CUDA.
- This is a correctness baseline, not yet an optimized concurrent implementation.

## Validation

`cargo test` checks geometry and preprocessing, and `cargo clippy --all-targets --
-D warnings` checks Rust code. Run `python -m pybaseline.rust_reference --help` for
the same-scope Python/PyTorch and Python/ORT reference pipelines. All three use
FP32, disabled TF32, detector side 1024 and recognition batch 128 in the first
comparison. Tests run sequentially in separate processes with at least 30 seconds
of measured work, on control, large-text and white-on-black pages.

`python -m pybaseline.rust_report` compares saved results and creates
`pybaseline/results/rust_slice/report.html` and `comparison.json`.
Dependencies are locked by `Cargo.lock`, the existing Python baseline lock and
`pybaseline/requirements-rust.txt`. Model SHA256 values and normalization/vocabulary
metadata are recorded in `models/metadata.json`. Generated weights are gitignored.

`python scripts/validate_rust_exports.py` additionally checks dynamic detector
shapes and PARSeq batches 1 and 3 against export-mode PyTorch logits. Maximum
observed absolute differences in this check were below 0.000024.

Initial results: 1.84 pages/s in Rust, 1.49 in Python/ORT, and 1.47 in Python/PyTorch.
All three detected 1,957/1,957 labelled words and recognized the same 1,956 correctly.
Every native word matched both references at IoU 0.99 with identical text. These
three synthetic pages establish an initial parity check, not general OCR coverage.
