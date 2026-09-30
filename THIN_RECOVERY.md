# Thin-character recovery experiment

Status: **available in native Rust streaming as an opt-in setting**. Recovery is off by default.
Dense refinement now also reconciles mutually rejected seam detections. Word
splitting remains unchanged.

## Use

CLI: add `--thin-recovery` to the throughput/pipeline command. Python: rebuild
the wheel, then set `config["thin_recovery"] = True` before constructing `Stream`.
Omitting the setting keeps recovery off, including in old saved profiles.

Recovery reuses the detector's raw/opened masks and the existing recognition
queue. It works on full-page maps and, when enabled, dense-refinement tiles.
Accepted words include `thin_recovery` evidence (map source, raw component bounds,
and two supporting anchors). Those evidence coordinates belong to the detector
map; the word's quadrilateral is mapped back to the original image as usual.
Rejected candidates are omitted. Extra words are appended, without reordering
existing words; callers needing reading order should use geometry.

The page retains its existing admission credit throughout recovery. The limit is
32 added crops and one million source crop pixels per page. Line-guided rotation
has its own separate budget and does not act on recovery candidates. Neighbour
search uses vertical buckets. No additional detector pass or model is loaded.

## Current extension

The native flag now includes the frozen one-hop anchor rule: when exactly one
nearby word supports a thin component, one aligned word beyond that neighbour
can supply the second anchor. This is one hop only; all thresholds, recognition
filters and page budgets remain unchanged.

Across 23 variants, versus the previous native recovery: 39 additional words,
38 with correct text and at least 50% labelled-word coverage; one correct `I`
with insufficient box coverage. No previous word text or box changed. Ordinary
omissions fell from 80 to 41 (original set 60 → 37; holdout 20 → 4). The additions
are 38 `I` characters and the seam word `agree`. This is a synthetic sample,
not evidence of a general digit-recovery gain.

The seam fix runs with dense refinement, independently of the recovery flag.
Both tiles must support a box crossing their ownership boundary at IoU ≥0.5.
If both reject ownership, keep one by objectness unless it overlaps an existing
word; if both own it, remove the duplicate. Empty tiles retain the original
fallback behavior. Recovery proposals themselves retain their existing tile
ownership filter.

Four warmed 230-page old/new/new/old trials (920 measured pages) gave pooled
2.459 pages/s before and 2.466 after (+0.26%, effectively unchanged at this sample
size). New runs peaked at 5.86–5.87 GiB total device VRAM; their whole-run idle
increment was 4.24–4.25 GiB. Old peaks varied 5.99–6.22 GiB, so this does not
establish a memory saving. The three-page limit held; all repeated outputs were
stable. This remains a 6 GiB-arena desktop run, not a 4GB-device result.

29 Rust tests, 18 Python tests, Clippy, cached candidate parity and the rebuilt
installed-wheel producer/consumer check passed. Nine of the 39 additions also
pass exact text + strict IoU 0.5; most narrow glyphs still have loose boxes.

[Current native report](pybaseline/results/native_anchor_v1/report.html).
CPU parity: `python scripts/check_anchor_recovery.py` (saved caches and preserved
v1 probe required). Native comparisons use `scripts/benchmark_anchor_recovery.py`;
see [remaining-omissions notes](REMAINING_OMISSIONS.md).

## Initial direct-anchor result

| Corpus | Page variants | Ordinary omissions before → after | Correct additions | Other additions |
|---|---:|---:|---:|---:|
| Existing statements and dense layouts | 11 | 55 → 49 | 6 | 0 |
| Fresh numeric fields and prose | 6 | 14 → 11 | 3 | 0 |

All nine recovered words were `I`. Four pass strict IoU 0.5 matching; five have
the right text with looser boxes. No existing box or transcription was changed.
These are repeated variants of nine source layouts, not 17 independent documents.
The recovery rate is 9/69 omissions on this subset, **not** 9/320 from the earlier
35-page audit. Remaining omissions are unresolved.

The fresh fixture has Arial/Times layouts, each upright, skewed 0.6° and rotated
90°, with ordinary fields containing digits 0–9, sentences containing I/a, table
rules and barcode-like bars. Its 120 digit instances already have baseline
detections; some are misread or have loose boxes. It establishes neither a digit
recovery gain nor broad false-positive safety. The three additions were all on
the skewed Arial page; none covered the negative-control marks.

[Visual report and added crops](output/diagnostics/thin-recovery/index.html).
Raw results: `pybaseline/results/thin_recovery_v3/results.json` and
`pybaseline/results/thin_fresh_v1/recovery/results.json`.

## Rule tested

- Find narrow threshold-mask components completely discarded by 3×3 opening.
- Require two nearby wider components with compatible height and line position.
  The current version also allows the one-hop support described above.
- Add vertical crop context from those neighbours. Reject overlap with existing
  predictions and duplicate proposals; never split or replace an existing word.
- Bound proposals to 32 and added source crop area to one million pixels/page.
- Recognize using the existing PARSeq model; accept only one or two alphanumeric
  characters with confidence ≥0.9. No forced I substitution or language rules.

Rules were developed on the existing cached maps, then fixed before the fresh
fixture evaluation. Native detector maps are reused: the experiment does not
rerun detection to propose candidates. Full-page maps were added to the cache
utility for the fresh run; older caches have tile maps only.

The initial prototype used Python/OpenCV for diagnosis. Native Rust proposals
now match all nine saved prototype proposals across 17 pages. End-to-end native
inference also recovers all nine, preserving every existing text/box on this
sample. This remains a small synthetic accuracy sample.

## Initial direct-anchor throughput result

Four warmed 340-page runs, off/on then on/off, on the local RTX 5070 Ti:

| Mode | Trial 1 pages/s | Trial 2 pages/s | Pooled pages/s |
|---|---:|---:|---:|
| Recovery off | 2.380 | 2.380 | 2.380 |
| Recovery on | 2.371 | 2.374 | 2.372 |

Observed throughput cost: **0.32%**. This is a small difference from two trials,
not a statistical significance claim. Summed CPU postprocessing increased by
about 4.7 ms/page; most of that extra work overlapped with other pipeline stages.
Measured total device VRAM peaked at 5.82–5.83 GiB in both modes. Whole-run peak
increment above each run's idle was 4.38–4.50 GiB, including loading/warmup; this
was not a 4 GiB-budget test. Three-page admission remained bounded in every run.

All 1,360 measured pages drained in order with stable repeated text and f32
coordinates. Each enabled run added the same 180 recovered words (nine ×20).
These repetitions demonstrate stability, not additional independent accuracy cases.

[Native benchmark report](pybaseline/results/native_thin_v1/report.html) ·
[Repeated-output checks](pybaseline/results/native_thin_v1/repeat_checks.json).

## Reproduce

Requires the baseline environment, exported models and a built
`refinement_probe` binary. Use fresh output directories:

```powershell
.venv-baseline/Scripts/python.exe scripts/test_thin_recovery.py --source pybaseline/results/dense_tuning_v2 --output pybaseline/results/my_thin_existing --recognize
.venv-baseline/Scripts/python.exe scripts/prepare_thin_validation.py --output pybaseline/results/my_thin_fresh
.venv-baseline/Scripts/python.exe scripts/test_thin_recovery.py --source pybaseline/results/my_thin_fresh/baseline --output pybaseline/results/my_thin_fresh/recovery --single --recognize
.venv-baseline/Scripts/python.exe scripts/report_thin_recovery.py pybaseline/results/my_thin_existing pybaseline/results/my_thin_fresh/recovery
```

The experiment saves its script and candidate evidence with the results. Existing
cached experiments are prerequisites for the first command; the fresh fixture
commands can run independently. The report command is CPU-only.

Native benchmark: run VS Code task **OCR: benchmark native thin recovery**, or
`python scripts/benchmark_thin_recovery.py --pages 340` using the baseline Python
environment. It requires the saved dense-tuning and fresh-fixture manifests above.
It runs quality checks followed by off/on and on/off trials, 340 pages each.
Loading and one workload warmup are excluded; ordered output and drain are included.
Reports, per-run resources, binary hashes and quality evidence go to a timestamped
directory. Use `--quality-only` to skip sustained timing.

Historical direct-anchor parity: `python scripts/check_thin_recovery.py` uses the
preserved v1 executable. Current parity: `python scripts/check_anchor_recovery.py`.

Validation includes 27 Rust tests, 14 Python tests, candidate parity, native
end-to-end quality and the installed-wheel producer/consumer check. Clippy passes.

## Remaining limits

Follow-up: [the remaining-60 audit](REMAINING_OMISSIONS.md) records the diagnosis
that led to the native seam fix and one-hop extension.

Most omissions remain unresolved. Wider false-positive coverage is needed before
considering a default. Improving the five loose boxes is a separate quality task;
do not force thin glyphs into the IoU metric at the cost of readable crops.
