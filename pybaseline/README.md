# PyTorch docTR baseline

Run from the repository root in PowerShell. An NVIDIA CUDA GPU is required;
the harness refuses to silently fall back to CPU.

```powershell
./pybaseline/setup.ps1
./.venv-baseline/Scripts/python scripts/generate_test_pdfs.py
./.venv-baseline/Scripts/python -m pytest pytests -q
./.venv-baseline/Scripts/python -m pybaseline.matrix
./.venv-baseline/Scripts/python -m pybaseline.report
```

The original `.venv` is preserved. Setup uses `.venv-baseline`, CUDA 12.8 PyTorch
2.11.0, torchvision 0.26.0, docTR 1.1.0 and `requirements-lock.txt`. A compatible
NVIDIA driver, `uv`, internet access and several GB of disk are needed. A separately
installed CUDA toolkit is unnecessary for these inference wheels. Official docTR
weights download to `.cache/doctr`.

## Corpus

The generator creates a 12-page PDF, RGB PNGs and a manifest with 17,019 word labels.
It uses deterministic seeded Lorem ipsum and built-in Helvetica, Times, Courier,
bold and italic fonts. Cases: A4 ordinary/large/small text, dense columns, tight/loose
tracking and leading, white-on-black, font blocks, 7-degree skew, mixed 0/-12/90/-90/180
degree text, and one very dense four-column A3 page with 5.5-point text.

Labels come from drawing commands, not OCR or PDF word extraction. Polygons use
normalized top-left coordinates, TL/TR/BR/BL in each word's own orientation. Bounds
use advance widths and ascent/descent, not tight ink. Italic overhang can affect
strict scores. All polygons are asserted on-page. Default rendering is 200 DPI;
`--dpi 300` is available. Use a new results directory when changing corpus settings.
Synthetic Latin text cannot establish accuracy on real scans or multilingual material.
The dense A3 page heavily weights micro-averages; inspect per-page results.

## Experiments

The matrix runs nine detectors, nine recognizers, four OCR pairs, then FAST-base/
PARSeq with crop orientation, page straightening, 1536 input, stretched aspect ratio,
larger batches and one PyTorch/OpenCV thread. Detectors use axis-aligned defaults;
rotated cases are deliberately difficult. Resolution/stretch variants are experiments,
not changes to docTR defaults. Cases run sequentially in fresh processes, recording
commands, statuses and failures. Completed results are skipped when resuming.
**Use a new output directory after changing code, settings or corpus**: resume does
not validate old results against newly supplied settings.

```powershell
./.venv-baseline/Scripts/python -m pybaseline.matrix --only reco_
./.venv-baseline/Scripts/python -m pybaseline.run --kind ocr --det fast_base --reco parseq --output pybaseline/results/custom --min-seconds 60 --repeats 5

# Additional diagnostic pass, excluded from throughput timing
./.venv-baseline/Scripts/python -m pybaseline.run --kind ocr --det fast_base --reco parseq --output pybaseline/results/profile --profile

# Synchronized process scaling; output directories must be new
./.venv-baseline/Scripts/python -m pybaseline.concurrency --workers 1 --output pybaseline/results/concurrency1
./.venv-baseline/Scripts/python -m pybaseline.concurrency --workers 2 --output pybaseline/results/concurrency2
```

## Timing and accuracy contracts

Float32, eval/inference mode, no AMP/compile, fixed seed, four PyTorch/OpenCV threads,
page/detector batch 2 and recognizer batch 128. docTR's own thread pool is independent
of `--threads` (up to 16 workers here). One full-corpus warmup supplies predictions
for accuracy. Whole-corpus trials repeat until both three trials and 20 measured
seconds are reached, with CUDA synchronization at boundaries. Throughput = total
items / measured seconds. Percentiles are **corpus-pass latency**, not page latency.
Repeats improve timing duration, not accuracy diversity.

Detector timing includes preparation, transfers, inference and box postprocessing.
Recognition uses all ground-truth crops rectified upright with 2 pixels of context,
in outer chunks of at least 512 words and internal batches of 128. OCR includes
detection, cropping, optional orientation, recognition and document construction.
PDF rasterization, PNG loading, ground-truth crop preparation, model loading/download,
scoring, serialization and disk I/O are excluded. Loading/preparation times are
recorded separately. This is warm inference throughput, not request-to-JSON latency.

- Polygon IoU >= 0.5 with maximum-cardinality one-to-one matching, independent of text.
- Detection precision/recall/F1 are micro-averaged across labels.
- OCR exact recall requires a matched box and case-sensitive exact word.
- Recognition reports exact/casefold accuracy and CER (summed Levenshtein edits /
  ground-truth characters). It does not evaluate detection or orientation classification.
- Merge candidates cover >=50% of multiple labels. Split candidates have multiple
  predictions each >=50% inside one label. These are geometry heuristics, not
  definitive semantic error counts.

No arbitrary 1-to-10 ratings: the report ranks measured accuracy alongside speed/memory.

## Monitoring and profiling

An independent process samples psutil and NVML every requested 50 ms (`--sample-ms`).
`telemetry.jsonl` contains nanosecond timestamps, process CPU%, system CPU per logical
core, RSS, device GPU activity, memory-controller activity, VRAM, power and SM clock.
Actual intervals are visible in timestamps. NVML's sensor window can be much slower;
faster polling does not create more hardware samples. 100% CPU means one logical core.
GPU utilization is time with a kernel active, not percent peak FLOPS. Device values
include the desktop and other applications. Torch peak allocated/reserved bytes are
process allocator values and omit some CUDA context/library allocations. WDDM
per-process VRAM is not assumed available. Missing sensors are null, not zero.

HTML embeds GPU/CPU timelines. `--profile` adds `cpu.prof`, `cpu_profile.txt`,
`stages.json`, `torch_profile.txt` and a Chrome/Perfetto-compatible `trace.json` in
separate diagnostic passes. cProfile cumulative times overlap; do not sum them.
Native work/waiting is attributed to Python callers, and profiling adds overhead.
CUDA events may need CUPTI and can be unavailable on Windows; inspect warnings.

Concurrency workers warm up before a shared gate. Aggregate throughput uses earliest
measured start/latest end; start spread is recorded. Device metrics are shared:
never sum device VRAM or GPU activity. Run these after the isolated matrix.

## Provenance

Each configuration stores versions, hardware, corpus SHA256, individual timings,
per-page scores, predictions and raw telemetry. Loading includes download time when
uncached, so it is not a clean cold-start comparison. Dependencies are pinned in
`requirements-lock.txt`; docTR checks its configured weight hash prefix. Generated
artifacts and caches are gitignored. The original broken `.venv` remains untouched.

The report works offline. To rebuild elsewhere use `python -m pybaseline.report
--input <matrix-folder> --output <report.html>`. Upstream observations refer to
tagged 1.1.0 source/release notes; historical versions were not benchmarked. These
measurements alone cannot prove the GIL is the dominant bottleneck. `/src/` is untouched.

Additional utilities:

```powershell
./.venv-baseline/Scripts/python -m pybaseline.inspect_environment
./.venv-baseline/Scripts/python -m pybaseline.validate_results
./.venv-baseline/Scripts/python -m pybaseline.export_telemetry
./.venv-baseline/Scripts/python -m pybaseline.sensitivity
# After profile and concurrency runs, add their comparisons to the report
./.venv-baseline/Scripts/python -m pybaseline.findings
./.venv-baseline/Scripts/python -m pybaseline.report
```

`UPSTREAM_REVIEW.md` records concrete changes between tagged 0.12.0 and 1.1.0 source.

See [STREAMING.md](STREAMING.md) for the single-character rotation corpus, model/policy
screen, batch tuning and three five-minute feeder/inference/writer runs.

## Controlled resolution study

Run `python -m pybaseline.resolution_study`, then `python -m pybaseline.resolution_report`.
The report is `results/resolution/report.html`. This uses the same 900-word upright
11-point A4 page at source DPI 100/200/400/600 and detector sides 1024–3072, with
FAST base and DB ResNet34 plus fixed PARSeq. A separate detector-only padding
ablation compares square and rectangular tensors containing identical text pixels.
The script resumes saved square cases: use a fresh results/resolution folder when
changing the experiment. Three timing repetitions are a short diagnostic, not a
sustained throughput benchmark. No tiling is implemented in this study.
