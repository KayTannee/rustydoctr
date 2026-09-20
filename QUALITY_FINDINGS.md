# Quality experiments: September 2026

Initial experiments were accuracy only while another application used the GPU.
Results: `pybaseline/results/quality/report.html`. The subsequent opt-in native
implementation and isolated benchmark are described in [DENSE_REFINEMENT.md](DENSE_REFINEMENT.md).

## Findings

Same DB ResNet34 + PARSeq FP32 weights; 300-DPI source pages. Exact words require
one-to-one matching at IoU >= 0.5 plus exact text. Labels use font metrics rather
than tight ink; isolated characters and punctuation can fail the overlap test.

| Statement | Truth words | Base 1536 | Selective dense rescan | Full 2048 | Full 2560 |
|---|---:|---:|---:|---:|---:|
| 0 | 353 | 277 | 287 | 292 | 294 |
| 1 | 462 | 363 | 385 | 381 | 381 |
| 2 | 519 | 404 | 437 | 432 | 438 |

- **Larger input can hurt large text.** In the size ladder, 96-pt `Sit` is correct
  at 1536 but loses its complete detection match at 2048/2560. Small legal text
  benefits strongly. This tests detector input size, not source DPI alone.
- **Selective rescans are promising.** Image-component density selects a small
  text band, then two overlapping 1024 tiles. No truth labels guide selection.
  This uses 4.46M detector input pixels versus 6.55M at 2560 (32% fewer), but
  slightly more than full 2048. Pixel count is not a runtime measurement.
- **Confidence misses segmentation errors.** `provided; I retain` becomes
  `provided;Iretain` with 99.996% recognition confidence. Use image/geometry
  evidence as well as confidence.
- **Coarse page direction works on these samples:** 27/27 correct using the CPU
  page classifier. Fractional skew correction helps the +/-0.5-degree examples,
  but sometimes hurts +/-1.5-degree cases. Keep float angles and require evidence
  before deskewing; ruled synthetic pages are an easy skew-estimation case.
- **Local rotation remains unresolved.** Changing crop orientation alone recovers
  none of the complete sidebar/diagonal words. Targeted region rescans recover
  four of seven sidebar words and the diagonal `Amount` on two pages, but a false
  diagonal candidate damages upright legal text on page 2: 404 -> 392 exact words.
  Do not enable this replacement policy by default.
- **I/underscore errors have multiple causes.** A tight upright underscore crop
  can become a dash without rotation. Preserve line direction and baseline context;
  a nonzero classification of a symmetric I or dash is not itself proof of error.

All three layouts are development fixtures, not held-out validation. The 256 and
512 rotated-region trials changed crop geometry too, so their difference cannot
be attributed solely to resolution. Raw predictions retain these failures.

## Next implementation

Keep the measured upright path. Add an opt-in page-direction/float-skew stage,
then bounded dense-region detection refinement. Reconcile boxes before final
recognition so we do not repeatedly recognize the entire page. Keep existing
queue credits and resident models; cap region count, area and retries.

Before shipping local rotation: require coherent direction evidence, protect
existing upright words, and log rejected as well as accepted replacements. Test
unseen layouts, sparse/unruled scans, opposite local directions and real scans.
Measure throughput only with the GPU free. Source-DPI-only and fixed full-width
band comparisons remain outstanding.

## Reproduce from the repository root

Use the existing `.venv-baseline` and exported models. Generate samples with the
VS Code task **OCR: generate quality sample pages (CPU only)**. Then run the
following sequentially; inference commands use the GPU but do not stress-loop:

```powershell
./.venv-baseline/Scripts/python -m pybaseline.quality_study --run-inference
./.venv-baseline/Scripts/python -m pybaseline.quality_orientation_cpu
./.venv-baseline/Scripts/python -m pybaseline.quality_experiments --run-inference
./.venv-baseline/Scripts/python -m pybaseline.quality_local_rotation --run-inference
./.venv-baseline/Scripts/python -m pybaseline.quality_rotated_regions --run-inference --size 512
./.venv-baseline/Scripts/python -m pybaseline.quality_report
```

The scale study reuses existing per-page JSON files. For changed fixtures,
weights or settings, archive the entire `pybaseline/results/quality` directory
first and run fresh. Other experiments require the scale study outputs and
overwrite their own results. The report rebuild itself uses no inference.
`rotated_roi_256_results.json` preserves an earlier exploratory implementation;
the current code does not exactly reproduce that historical geometry policy.
