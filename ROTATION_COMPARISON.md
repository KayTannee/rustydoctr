# Python docTR versus Rust: rotation comparison

Actual installed docTR 1.1.0 `ocr_predictor`, DB ResNet34 + PARSeq FP32, detector
1536, recognition batch 256, TF32 off. All 27 statement variants plus the size
ladder; same model weights and source pixels as the Rust accuracy study.

| Mode | Exact statement words / 12006 | Full-page detector passes |
|---|---:|---:|
| docTR crop rotation, no straightening | 8536 | 1 |
| docTR default straightening, upright crops | 9363 | 2 |
| docTR default straightening, crop rotation | 9141 | 2 |
| Rust coarse page direction | 9321 | 1 |
| Rust coarse + fractional deskew | 9346 | 1 |
| Rust coarse + deskew + dense refinement | 9989 | 1 plus tiles |

**Near parity for rotation alone:** Rust deskew is 17 words behind the best
tested docTR configuration, a 0.14 percentage-point difference. Fractional
angle precision has not established an OCR accuracy advantage. Rust avoids
the initial full-page detection used by docTR's straightening estimator; this
study counts passes but does not establish a runtime speedup.

**The I/dash problem is not just the classifier.** With docTR's same oriented
boxes and default page straightening, disabling the crop classifier changes
exact words from 9141 to 9138. Matched I-to-dash cases remain: 37 enabled versus
36 disabled. Geometry/cropping and recognition context need attention too.
Many single characters are unmatched or merged; the report includes those counts
so fewer matched errors cannot be mistaken for better recognition.

**A configuration trap in this installed version:** `preserve_original_coords`
selects a different raster straightening implementation. On statement 1 at 90
degrees it produces a 3508x4448 image with 44.7% black pixels, versus 3507x2481
and 0.9% for the default path. This changes text scale at the fixed detector size.
With upright crops, the preserve-coordinates path scores 8850, versus 9363 for
the default. The comparison therefore tests both, rather than penalizing docTR
for an option chosen just to simplify scoring.

The docTR default path returns rectified coordinates. An observational helper
reconstructs its exact padding/rotation/crop transform, verifies byte-identical
rectified pixels, and maps predictions back without changing inference.
A CPU landmark test verifies the inverse on quarter and small rotations.
For the preservation path, another observer retains quadrilaterals before docTR
reduces upright boxes to envelopes. Public-geometry scores are also reported.

Strict exact counts use one-to-one IoU >= 0.5 and exact text. Single-character
and local rotated-word diagnostics use IoU >= 0.25 because labels use font
metrics. None of the page-straightening configurations reliably recovers the
local sideways/diagonal content. Dense refinement is an additional segmentation
policy, not evidence that Rust rotation alone beats docTR.

## Run or inspect

Local report: `pybaseline/results/doctr_orientation_comparison_v1/report.html`.
Raw predictions, flags, source/model/image hashes, observed detector calls,
angles, confusion counts and coordinate overlays are alongside it.

```powershell
./.venv-baseline/Scripts/python -m pybaseline.compare_doctr_orientation
```

Runs seven docTR configurations sequentially and builds a fresh timestamped
report. Requires existing fixtures/models and the saved native reference at
`pybaseline/results/native_orientation_v1`; override with `--rust-results`.
Model and manifest hashes must match. The Python recognizer uses its native
decoder; Rust uses the exported graph. Upright control exact totals match.
Diagnostic timings are not a sustained throughput comparison.

Rebuild only the HTML, without inference:

```powershell
./.venv-baseline/Scripts/python -m pybaseline.doctr_orientation_report pybaseline/results/doctr_orientation_comparison_v1
```

Synthetic development fixtures only; validate on unseen real scans before
selecting defaults or claiming a general quality advantage.
