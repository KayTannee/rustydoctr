# Sustained pipeline and ambiguous-character tests

This extends, rather than replaces, the original 12-page / 28-configuration baseline.
All supporting code remains outside `src/`.

For a quick GPU-Z observation run in VS Code: **Terminal → Run Task →
docTR: streaming throughput (GPU-Z)**. Select a page and 300 or 600 seconds.
This uses DB ResNet34 + PARSeq at 1536 with batches 8/512, with a fresh timestamped
output directory under `pybaseline/results/manual_streaming` each time. Loading
and warmup occur before the timed feed; queued pages drain afterward. The launcher
is `pybaseline/run_throughput.py`; the actual pipeline is `pybaseline/stream.py`.

```powershell
./.venv-baseline/Scripts/python scripts/generate_rotation_pdfs.py
./.venv-baseline/Scripts/python -m pybaseline.rotation_screen
./.venv-baseline/Scripts/python -m pybaseline.stream_suite
./.venv-baseline/Scripts/python -m pybaseline.validate_streams
./.venv-baseline/Scripts/python -m pybaseline.rotation_evidence
./.venv-baseline/Scripts/python -m pybaseline.extension_report
./.venv-baseline/Scripts/python -m pybaseline.report
```

The generator creates three 250-DPI A4 pages and 846 word labels: upright, 7-degree
skew, and skew with genuinely rotated side text. Context includes “It was cold outside
I put my coat on”, repeated `I`, `i`, `l`, `1`, `_`, `-`, and other single characters
in three fonts. Word labels include their sentence, font, global/local angles and
whether they belong to the rotated region. Full-page PNGs, a contact sheet, PDF and
manifest are saved in `testdata/rotation`.

## Accuracy selection

Five model pairs are screened with per-crop orientation enabled at detector size 1536.
The leading pair is then tested with straight-page assumptions, page straightening
with crop orientation retained, and page straightening with crop orientation disabled.
This last case is an ablation of local orientation, not an exact reproduction of an
external two-full-OCR-pass application workflow. Native `straighten_pages` already
performs a detector pass before and after page correction; `preserve_original_coords`
maps exported geometry back to the original page for scoring.

Selection uses IoU-0.5 exact-word recall, with singleton accuracy and throughput as
tie breakers. The selected streaming policy must retain crop orientation to support
genuinely rotated words. This small synthetic development set is not held out, and
the selected pair is provisional. Compare rotated-word scores separately: a model
with the best aggregate may be worse on the small rotated subset.

Character diagnostics use one-to-one IoU-0.25 matching, with explicit confusion counts
and original sentence context. Font-metric boxes are a poor fit for dash/underscore
ink, so unmatched punctuation is not automatically a classifier error. The concrete
contextual-I artifact includes the exported crop orientation/confidence and an image
overlay. It helps separate output symptoms from speculative root causes.

## Streaming contract

Three work processes form a bounded pipeline:

1. Feeder: decode the PNG anew for each page from the warm filesystem cache, convert
   BGR to RGB into a shared-memory slot, enqueue an identifier/slot/timestamp.
2. Inference: fill a page batch, run the complete unmodified docTR OCR predictor,
   synchronize CUDA, export the document, release input slots, enqueue exported pages.
3. Writer: validate sequence numbers, serialize all word geometry/text to JSONL,
   flush each batch, and perform a final `fsync` before completion.

The input pool holds twice the page batch. The output queue holds two batches.
Backpressure prevents unbounded memory growth. Pixels do not get pickled through
the input queue; exported result dictionaries do cross a process boundary. There is
one inference process, not multiple copies of the GPU model. FP32, eager PyTorch,
one PyTorch/OpenCV intra-op thread and docTR's own thread pools are used.

Two full batches warm the model before measurement. Loading/download/warmup are
excluded; worker startup, decode/copy, complete OCR, export/IPC, serialization, writing,
final fsync and drain are included. This is best-case local raster input, not PDF
rasterization, network storage or a database postprocessor. Repetition gives a
sustained workload, not additional accuracy diversity.

The suite tests page/recognition batch pairs 1/128, 4/256, 8/512 and 16/1024 for 20-second
feeding windows. It chooses the fastest completed pipeline run that preserves the
highest warmup-sample exact count. This is a bounded tuning sweep, not a global optimum;
short-run startup/drain effects are visible. It then feeds each of the three pages
for 300 seconds and drains queued work, so each long run lasts slightly over five
minutes. `--seconds 600` requests ten-minute feeds.

```powershell
# Fresh output directories required for individual runs
./.venv-baseline/Scripts/python -m pybaseline.stream --selection pybaseline/results/rotation_screen_v2/selection.json --page skew7_mixed_characters --batch 8 --reco-batch 512 --seconds 300 --output pybaseline/results/my_stream

# Use a fresh suite directory for changed settings
./.venv-baseline/Scripts/python -m pybaseline.stream_suite --output pybaseline/results/another_stream_suite --seconds 600
```

Each run saves `summary.json`, per-batch completion times, feeder/writer stage and wait
times, page latency percentiles, warmup accuracy sample, 50-ms-requested NVML/CPU logs,
and **every exported page** in `pages.jsonl`. Latency is admission before decode through
writer flush; time blocked before admission is backpressure, not part of page latency.
Aggregate throughput includes that bounded pipeline's complete measured makespan.
Stage totals overlap across processes and must not be summed. The continuous CPU
sampler measures the inference process; total pipeline CPU cores are derived from all
three process CPU-time totals. Device GPU/VRAM still include other applications.

Reports can be rebuilt while or after cases complete. Incomplete runs are never shown
as completed results. Generated datasets/results are gitignored. Do not change code,
corpus or settings and resume into old completed results: use a fresh output location.
