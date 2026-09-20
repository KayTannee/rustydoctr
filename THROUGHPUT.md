# Bounded word-OCR throughput experiment

For the short setup/profile workflow, see [BENCHMARK.md](BENCHMARK.md).
For the installable wheel and producer/consumer example, see [PYTHON.md](PYTHON.md).

Use **Terminal → Run Task → Rust: bounded auto-tuned throughput (GPU-Z)**.
Choose 300 or 600 seconds. Startup calibration and model warmup are additional
time; each calibration trial has its own process and full-corpus warmup. Results
go to `pybaseline/results/throughput/watch-<timestamp>/run/summary.json`, with raw
telemetry and ordered word predictions alongside it. The original vertical-slice
task remains available for its smaller, older experiment.

The selected model pair is DB ResNet34 + PARSeq, at detector resolution 1536,
using the same pretrained weights as the earlier accuracy study. This is an
upright word pipeline: PNG decode, detector, CPU geometry/crops, recognizer,
JSONL output. Rotation, layout reconstruction, retries, PDF rasterization and
quality improvements are deliberately outside this throughput comparison.

## Reproduce

After the existing Python baseline/corpus and Rust model export setup:

```powershell
./.venv-baseline/Scripts/python.exe scripts/prepare_throughput.py
$env:CARGO_HOME = "$PWD/.cache/cargo"
& "$env:USERPROFILE/.cargo/bin/cargo.exe" build --release --locked

# Automatic calibration followed by five measured minutes (plus queue drain).
./scripts/run_pipeline.ps1 --auto --seconds 300 --output pybaseline/results/throughput/my-auto

# Fixed configuration; useful for repeat measurements without recalibration.
./scripts/run_pipeline.ps1 --det-batch 1 --reco-batch 512 --inflight 4 --workers 2 --arena-mib 8192 --seconds 300 --output pybaseline/results/throughput/my-rust

# Python references. Use fresh output directories for every run.
./.venv-baseline/Scripts/python.exe -m pybaseline.throughput_python --backend torch --page-batch 4 --reco-batch 1024 --seconds 300 --output pybaseline/results/throughput/my-torch
./.venv-baseline/Scripts/python.exe -m pybaseline.throughput_python --backend ort --page-batch 1 --reco-batch 512 --seconds 300 --output pybaseline/results/throughput/my-ort
```

Run GPU benchmarks one at a time. The repetitions include ten upright synthetic
pages (16,246 labelled words), including dense A3. Inputs are reread/decoded on
every pass; the operating-system file cache is warm. Each run warms the complete
corpus before timing. Timing includes queue startup/drain and output flush/fsync,
and stops admitting work at a corpus boundary after the requested duration.
`--pages N` instead processes an exact number of pages, for calibration/testing.

The Python/Torch reference uses native docTR modules and its normal early-ending
PARSeq decoder. Python/ORT and Rust share the exported full-length decoder graphs.
Python prefetches two batches but runs OCR stages serially. Compare both; the
results measure implementations and scheduling, not the GIL in isolation.

## Bounded pipeline

`src/pipeline.rs` exposes reusable `sessions` and `run` functions. The CLI and
telemetry/calibration live in `src/bin/throughput.rs`.

1. The reader must acquire a page admission credit before decoding/preparing it.
2. A detector worker owns one CUDA session. A bounded queue feeds CPU box/crop
   workers while subsequent inference can proceed.
3. CPU workers prepare bounded crop chunks. The recognizer combines crops across
   admitted pages into batches, using a separate model session. Partial batches
   flush after a 5 ms input idle interval to avoid deadlocking small/blank pages.
4. The writer restores input order, flushes the record and returns its credit.

Prepared pages, detector results and crop chunks each have queue capacity two.
There is one model pair, not one copy per worker. On an error, all blocking stage
operations observe cancellation; the process fails instead of silently dropping
pages. Output sequence/count checks detect lost or duplicate results.

Automatic page slots estimate memory from the largest source image and detector
tensor, using `--host-mib 512` by default and at most eight admitted pages. Workers
default to at most two. This bounds work in progress; it is not a hard process-RSS
limit. `--inflight`, `--workers`, and `--host-mib` permit explicit adjustments.

Automatic CUDA arena budget is 60% of initially free VRAM, capped at 8 GiB, split
25% detector / 75% recognizer. `--arena-mib` overrides it. Runtime/model allocations
outside the arenas mean this is not a hard total-VRAM limit. Calibration uses
fresh subprocesses, checks sampled device-memory growth against the budget, and
rejects failed candidates. It chooses the least-memory eligible candidate within
3% of the fastest measured throughput, preserving the smallest successful
candidate's word texts on the calibration corpus. Use `--throughput-tolerance 0`
for fastest-only selection. Short trials cannot predict every later cache growth.
It does
not guarantee accuracy on unseen pages, dynamically retune an active workload,
or prevent every transient OOM. It requires functioning NVML for its memory gate.

## 4 GB target preset

VS Code: **Rust: 4 GB target throughput (GPU-Z)**, or:

```powershell
./scripts/run_pipeline.ps1 --vram-4gb --seconds 300 --output pybaseline/results/throughput/my-4gb
```

This flag automatically calibrates recognition batches 32, 64 and 128 with one
detector page per batch and at most two admitted pages by default. Both models
stay loaded in CUDA sessions throughout warmup and measurement. It preserves
the FP32 weights and selected detector resolution; no quantization, model swaps,
whole-model CPU fallback, or automatic quality changes.

The application target is the smaller of 3.5 GiB and 90% of currently free VRAM.
The nominal 4 GiB profile therefore leaves 0.5 GiB for non-OCR use. Another
0.5 GiB inside the application target is reserved for allocations outside the
CUDA arenas. With sufficient free memory, arenas total 3 GiB: 2 GiB detector and
1 GiB recognizer. Less free memory reduces those budgets. Explicit
`--det-arena-mib` allows a different allocation split.

A 100 ms sampler cancels work if incremental device memory exceeds the target;
it also stops if memory telemetry becomes unavailable. Arena limits control
most allocations, but sampling cannot prevent every transient overshoot or
driver paging. A failure leaves `failure.json` and telemetry instead of a
successful benchmark summary. If no candidate fits, the program fails clearly
rather than unloading one model to make room for the other.

The desktop has roughly 2.2–2.4 GiB of idle GPU allocations, so GPU-Z's total
can exceed 4 GiB during a desktop proxy test even when OCR's incremental usage
fits its target. A real 4 GiB card with that same idle usage has much less free
memory: the flag reduces the budget accordingly, and this model pair may then
fail to fit. The preset is not a claim that every 4 GiB GPU can run these models.
The laptop's architecture, available memory and driver behavior still need an
actual hardware test. Device-wide measurements can also include changes from
other applications during a run.

## Measurements

`summary.json` includes admitted-page high-water mark, latency percentiles, stage
work/wait times, throughput and resource summaries. `telemetry.jsonl` records
NVML utilization/device-wide VRAM and native process CPU/RSS. CPU 100% represents
one logical core. Sampling intervals are 100 ms Rust / 50 ms Python; NVML's sensor
updates more slowly. Stage wall times overlap and must not be summed. Prefer
pages/second at acceptable memory use over maximizing the GPU-Z percentage.

Generate the audit/report after completed runs:

```powershell
./.venv-baseline/Scripts/python.exe -m pybaseline.throughput_report --runs pybaseline/results/throughput/my-torch pybaseline/results/throughput/my-ort pybaseline/results/throughput/my-rust
```

The report validates every output's sequence/corpus index, compares first-pass
word accuracy and strict cross-backend box/text parity, and shows minute rates
and memory trends. Dense A3 accuracy is not an acceptance target at this stage;
small geometry differences are recorded rather than interpreted as a quality
improvement. Full model hashes and comparison data are retained in JSON.

## Adding CPU quality stages later

Extra CPU work can preserve steady-state pages/second if it finishes ahead of
the GPU's demand and stays within the bounded working set. Per-page latency can
still increase. Once CPU workers become the slowest stage, throughput falls;
the existing queue wait and admission metrics expose that change. Additional
model passes (for example, rotation retries) still add GPU work and cannot be
hidden as CPU work. Python can also be pipelined; this experiment proves the
current implementations' difference, not that Python is incapable of overlap.
