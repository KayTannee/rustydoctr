# Public labelled OCR benchmarks

All three datasets are downloaded under ignored `testdata/public/`. The first
comparison uses all 50 FUNSD, 100 CORD v2 and 1,634 HierText test images.
Training/development splits are retained separately; no test-set tuning is performed.
Sources checked 2026-09-23.

| Dataset | Why use it | Comparison target |
|---|---|---|
| [FUNSD](https://guillaumejaume.github.io/FUNSD/description/) | Noisy scanned forms with word boxes and text; closest small document test | docTR detection, recognition and end-to-end metrics |
| [CORD](https://github.com/clovaai/cord) | Indonesian receipts, prices and compact columns; published release has 800 train / 100 validation / 100 test images | docTR results, with exact annotation release and split pinned |
| [HierText](https://github.com/google-research-datasets/hiertext) | 11,639 scene/document images with word polygons, text and line context; larger independent stress test | Official word detection/end-to-end evaluator and published competition results |

FUNSD exposes word-level labels inside form entities: use these, not entity boxes.
CORD's public versions differ in annotation corrections; use the version expected
by the pinned docTR loader for reproducing its table. HierText has legibility /
do-not-care rules and requires mask_stride=1 for publication comparisons.

## Published reference values

[docTR's model table](https://mindee.github.io/doctr/latest/using_doctr/using_models.html)
reports the following percentages:

| Task | Model | FUNSD | CORD |
|---|---|---|---|
| Detection recall / precision | DB ResNet34, 1024 input | 82.76 / 76.75 | 89.20 / 71.74 |
| Recognition exact match on supplied word crops | PARSeq | 88.53 | 95.56 |
| End-to-end recall / precision | DB ResNet50 + PARSeq | 73.52 / 76.27 | 85.91 / 80.13 |

The end-to-end row uses **ResNet50**, unlike our ResNet34. The recognition row
assumes known word crops and is not whole-page accuracy. Published docTR scores
combine training and evaluation subsets; test-only runs are separate results.
The table does not pin all checkpoint/evaluation provenance, so treat figures as
reference values until reproduced with the matching version and settings.
Its timing uses different hardware and is not a Rust speedup baseline.

## Comparison protocol

1. Freeze versions, checkpoint hashes, annotation release and split manifests.
2. Run stock PyTorch docTR at 1024 and at our matched 1536 settings, Rust with
   enhancements off, then Rust with deskew/dense refinement/recovery on. Use the
   same DB ResNet34/PARSeq weights and original images. Add ResNet50 only for a
   separate reproduction of the published end-to-end row.
3. Use docTR's [LocalizationConfusion and OCRMetric](https://mindee.github.io/doctr/latest/modules/utils.html#task-evaluation)
   for its reference comparison: IoU >=0.5, exact text, recall / precision / F1.
   Separately test recognition on ground-truth crops, with the loader's exact
   text filtering/normalization recorded. Our relaxed 50%-coverage diagnostic
   does not replace these strict scores.
4. Report single-character recall, digits, missed words, introduced false
   positives, and loose boxes alongside the aggregate. Never drop hard short
   words from our diagnostic just because another benchmark ignores them.
5. Keep test pages frozen; tune on training/development pages only. Report any
   known checkpoint training overlap, and do not call combined-split results a
   held-out generalization test.
6. Add small-skew / 90-degree / 180-degree variants as a separate robustness
   suite with transformed labels. They cannot be compared directly to published
   original-image scores. Do not label upscaled scanned pixels as new 300-DPI
   source detail.
7. Measure warmed end-to-end throughput on the same hardware, reporting both
   total and idle-relative VRAM, batch/admission limits and output drain.

For HierText, use its official evaluator and ignore-region behavior rather than
substituting our synthetic scorer. Its word-level task can be evaluated without
adding layout or table inference to the Rust library.


## Run locally

Requires the existing baseline environment, exported models and release
`throughput.exe`. Install the two extra reader/evaluator dependencies once:

```powershell
uv pip install --python .venv-baseline/Scripts/python.exe -r pybaseline/requirements-public.txt
.venv-baseline/Scripts/python.exe scripts/download_public_datasets.py
.venv-baseline/Scripts/python.exe scripts/prepare_public_datasets.py
.venv-baseline/Scripts/python.exe scripts/benchmark_public.py
.venv-baseline/Scripts/python.exe scripts/score_public.py
.venv-baseline/Scripts/python.exe scripts/public_inventory.py
.venv-baseline/Scripts/python.exe scripts/report_public.py
```

VS Code task **OCR: report public test sets** runs download → preparation →
benchmark → scoring → HTML reporting (plus the Rust build). Run the inventory
command separately to refresh its download evidence. Downloads resume partial
files; completed inference/scoring runs are skipped. For changed OCR code or
settings, pass a new `--output` to the benchmark and matching `--root` to the
scorer/report scripts rather than reusing old scores.

[Report](../../pybaseline/results/public_v1/report.html) and raw JSONL/JSON results live
under `pybaseline/results/public_v1`. Six modes: stock Python and plain Rust at
1024 and 1536, stock Python rotation/straightening at 1536, and Rust enhanced at
1536. No additional table/layout models run. Published recognition-only figures
are context, not scores measured by this whole-page run.

Data pins: CORD v2 revision `7f0115a4b758a71d6473b8d085751692da2fef98`;
HierText source/evaluator revision `70b6620b2b112597d8219e11eee9773a1403827c`.
FUNSD and CORD downloads verify upstream SHA-256 checksums. HierText archives
record their downloaded SHA-256 values and extraction/split counts. Do not
commit the downloaded datasets. Upstream licences remain with their sources.

The HierText scorer calls the original evaluator directly, without Apache Beam's
CLI orchestration. No word polygon downsampling occurs. Adapter tests cover
ignored regions and real annotation parsing; `scripts/verify_public_evaluator.py`
reproduces every published sample word-detection score on all 1,724 validation
images. The official sample contains degenerate polygons that its evaluator logs
and handles; those have not been silently removed or relabelled.

Timings exclude model loading and full-corpus warmup, and include decoding and
result output. Runs here are single measurements, with downloads/scoring possibly
concurrent, so use them diagnostically rather than claiming controlled speedups.


## Stock rotation-path failure and separate compatibility diagnostic

The stock rotation-enabled docTR run hit a negative-stride NumPy crop error on
4/100 CORD test pages and 127/1,634 HierText test images. Its original results
keep those pages as explicit failures/empty predictions; no failed page is
removed from the denominator.

`scripts/recheck_public_rotation.py` separately retries those pages after applying
`copy(order="C")` to recognition crop arrays. This changes storage strides, not
pixel values, OCR models or thresholds. All 131 failed pages completed; three
previously successful controls per dataset retained identical text/geometry.
The separate `python_rotated_copy_1536` rows combine those repaired outputs with
the unchanged successful original outputs. They therefore have no full-run
throughput result and must not be presented as unmodified stock docTR.

Scoring uses all test pages for both variants. The original crash records and
per-page repair evidence are retained alongside the results.


## Completed results — 24 September 2026

Downloaded all splits (12,838 images), evaluated all 1,784 test images in six
configurations: 10,704 measured image passes, plus full-corpus warmups.
Exact end-to-end word F1 (%; box/polygon matching as documented above):

| Configuration | FUNSD | CORD v2 | HierText |
|---|---:|---:|---:|
| Python plain 1024 | 72.49 | 76.96 | 39.84 |
| Rust plain 1024 | 72.50 | 76.92 | 39.87 |
| Python plain 1536 | 72.89 | 69.19 | 44.68 |
| Rust plain 1536 | 72.92 | 69.23 | 44.66 |
| Python rotation 1536 (failures retained) | 67.98 | 68.04 | 47.40 |
| Python rotation + array-copy diagnostic | — | 65.85 | 48.90 |
| Rust enhanced 1536 | 72.39 | 58.52 | 42.32 |

Plain Rust preserves accuracy closely. On HierText its observed throughput is
12.43 vs 7.50 pages/s at 1024 (1.66×), and 8.40 vs 5.31 at 1536 (1.58×).
This changes both language and inference backend (ONNX Runtime vs PyTorch);
it is not evidence that language alone causes the gain. Timing limitations above apply.

Higher detector resolution is not universally better: it helps HierText, barely
helps FUNSD, and hurts CORD. The enhanced bundle regresses on every corpus.
Its 39 thin-recovery additions include only four label-verified coverage matches
and three strict matches. Some visible words may be unlabelled, but these results
do not support enabling recovery generally. No production defaults were changed.

Next: use training/development splits for separate orientation, deskew, dense
refinement and recovery ablations. Prioritise the HierText rotation gap and CORD
false positives. Keep this test result frozen and avoid tuning directly on it.
Published results remain contextual comparisons, not matched-model reproductions.

To reproduce the optional compatibility diagnostic after stock inference:

```powershell
.venv-baseline/Scripts/python.exe scripts/recheck_public_rotation.py
.venv-baseline/Scripts/python.exe scripts/score_public.py
.venv-baseline/Scripts/python.exe scripts/report_public.py
```

Resolution, tile overlap and heatmap-fusion accuracy experiments:
[TILING_ACCURACY.md](TILING_ACCURACY.md).
