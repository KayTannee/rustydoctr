# Experimental dense-text refinement

Opt-in, upright pages only. It keeps the existing detector/recognizer sessions
and bounded queues. A page keeps its admission credit through both extra tiles.
Default OCR behaviour is unchanged when the flag is omitted.

## Run

Build with `cargo build --release --bin throughput` (use the repository Cargo
cache as in `scripts/setup_rust.ps1`). Add `--dense-refine` to the existing command:

```powershell
./scripts/run_pipeline.ps1 --workload testdata/throughput.json --output pybaseline/results/my-refined-run --pages 192 --size 1536 --dense-refine
```

For Python, rebuild with `./scripts/build_python.ps1`, then set
`config["dense_refine"] = True` before constructing `Stream`. See [PYTHON.md](PYTHON.md).

Generate the quality fixtures first, then run VS Code task
**OCR: benchmark native dense refinement**, or:

```powershell
./.venv-baseline/Scripts/python scripts/benchmark_refinement.py --pages 192
```

The benchmark compares base 1536, refinement and full-page 2560 twice in reversed
order. Each trial warms its own resident sessions, then processes 192 pages
(three statement layouts repeated 64 times). Loading/warmup are excluded;
decode, OCR, ordered JSONL output and drain are included. Run with the GPU free.
Results, hashes and an HTML report go into a fresh timestamped results directory.

## Policy and limits

- Image-only connected-component density selects one small-text band. No truth
  labels, expected wording or recognition confidence enter selection.
- Skip bands taller than 25% of the page. Otherwise use two overlapping
  half-width crops, each detected at 1024. No recursive refinement.
- Merge detections before recognition. Crop centers determine ownership at the
  seam; original words outside the selected ownership region survive. Empty
  replacement detections retain the originals.
- Emit `refinement_tiles` with source-pixel bounds and ownership intervals for
  inspection. Old configuration JSON remains valid; the flag defaults to false.
- Both models stay resident. Refinement tensors and probability maps add bounded
  host buffers. Automatic admission estimates include them, but this is an
  estimate, not a byte-level memory guarantee. Explicit `inflight` overrides it.

These are 300-DPI synthetic development fixtures. Fixed source-pixel thresholds
need validation at other DPIs. This policy does not solve rotation, overlapping
multi-directional text, noisy scans, or every segmentation error. Region
replacement may lose individual words even when aggregate accuracy improves.
Do not assume previous 4-GB profiles remain valid without remeasurement.

## Validation

RTX 5070 Ti, DB ResNet34 + PARSeq FP32, two 192-page trials per mode:

| Mode | Combined pages/s | Exact words / 1334 | Incremental peak VRAM |
|---|---:|---:|---:|
| Base 1536 | 4.08 | 1044 | 4.37-4.38 GiB |
| Selective refinement | 3.33 | 1109 | 4.38 GiB |
| Full-page 2560 | 2.91 | 1113 | 4.50-4.51 GiB |

Combined throughput is total pages divided by total measured wall time. Base
trials were 3.95/4.22 pages/s; refinement 3.34/3.32; full 2560 2.89/2.93.
Refinement was 18% lower throughput than base, but 14% faster than full 2560.
It added 65 exact words over base and had four fewer than full 2560 across the
three unique pages. All 1152 measured copies retained identical word text within
their respective trials. Every run stayed within three admitted pages.

Both models were resident, with 6144 MiB total CUDA arenas (4096 detector),
recognition batch 256, detection batch 1 and two CPU postprocessing workers.
GPU utilization averaged about 91-93%. Memory is device-wide sampled peak above
pre-load idle, including warmup; this does not certify operation on a 4-GB GPU.
Raw results: `pybaseline/results/native_refinement_v1/`, including an HTML report,
per-page JSONL, 100-ms telemetry and binary/model/manifest hashes.

Native smoke results match the Python prototype: 287 / 385 / 437 exact words,
versus 277 / 363 / 404 at base 1536. The size ladder selects no refinement and
retains 14 strict exact matches. Scores require IoU >= 0.5 and exact text.

Rust tests cover empty/solid images, bounded region selection, seam ownership,
outside-word preservation and empty-rescan fallback. The installed-wheel check
feeds RGB pages independently of result consumption with one admission slot,
compares native word text/tile coordinates and checks clean shutdown:

```powershell
./.venv-baseline/Scripts/python scripts/check_refinement_stream.py
```

That integration check uses the current session's native smoke outputs under
`pybaseline/results/native_quality_smoke` and `.cache/quality-workload.json`.
It passed with the rebuilt, installed wheel: all four pages matched native word
text and tile coordinates, including the ladder with no refinement. Twelve Rust
tests and Clippy with warnings denied also passed.
