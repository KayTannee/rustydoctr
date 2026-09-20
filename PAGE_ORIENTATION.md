# Opt-in page direction and fractional deskew

Export the CPU page classifier once, then build Rust (or rebuild the Python wheel):

```powershell
./.venv-baseline/Scripts/python scripts/export_orientation_model.py
./scripts/build_python.ps1
```

The export produces `models/page_orientation.onnx` and `page_orientation.json`.
It uses the cached docTR MobileNet page classifier and validates CPU ONNX logits
against Torch. Copy both files alongside the detector/recognizer when deploying.
`scripts/setup_rust.ps1` now exports these files too.

## Enable

Native streaming CLI: add `--page-orientation --deskew` to
`scripts/run_pipeline.ps1`. Omit `--deskew` for coarse 0/90/180/270 correction only.
`--dense-refine` can be combined with either mode.

Python uses the existing producer/consumer API:

```python
config["page_orientation"] = True
config["deskew"] = True
with Stream(models="models", config=config) as ocr:
    ...  # submit_rgb on the producer; consume results independently
```

Both options default to false in old configurations. Deskew requires page
orientation. One CPU orientation session lives alongside the two GPU sessions;
the reader corrects pages before detector preprocessing. Admission and result
queues retain their existing bounds. Model sessions persist through warmup.

## Policy

- Classifier confidence below 0.9 leaves the page unchanged and skips deskew.
- Quarter turns are exact pixel permutations, with no interpolation.
- Fine skew uses horizontal ink-boundary projection scores at a maximum 1000-pixel
  analysis dimension. Search is +/-3 degrees, initially at 0.1-degree intervals,
  refined at 0.02-degree intervals. Values remain floating point.
- At least two horizontal bands, and a strict majority of evidence-bearing
  bands, must agree within 0.2 degrees. Require a 5% total score improvement and
  3% improvement in each agreeing band. Skip angles below 0.2 degrees and estimates
  near the search boundary. These are provisional conservative thresholds.
- Applied fine skew uses one expanded-canvas bilinear warp with a white border.
  Do not treat its angle as more precise than the image evidence/search supports.
- No full-page OCR rerun. Local 45/90-degree words and isolated-character
  orientation remain unresolved; they are not independently rotated by this mode.

## Coordinates and diagnostics

With page orientation enabled, each word contains:

- `quadrilateral`: four normalized corners on the **original input image**, in
  corrected reading-frame corner order. Corners can extend outside the source
  at padded edges; they are not individually clamped.
- `polygon`: the existing two-corner field, now the clipped axis-aligned envelope
  on the original image. Use the quadrilateral for precise overlap or overlays.

Page `geometry` records original/corrected sizes, classifier angle/confidence,
applied quarter turn, estimated/applied fine skew, evidence and abstention reason.
Angles describe clockwise input orientation, cancelled by counterclockwise image
correction. `corrected_to_original` is a 2x3 affine map of **pixel-edge** coordinates.
`refinement_tiles` are in corrected-image pixels; use that map when displaying
them over the original. `stage_seconds.page_orientation` separates CPU orientation
work from decode/preparation. Stage totals overlap and are not additive wall time.

## Validate

VS Code task **OCR: test native page orientation**, or after building and generating
the quality fixtures:

```powershell
./.venv-baseline/Scripts/python scripts/check_page_orientation.py --output pybaseline/results/my-orientation-check
```

The four sequential modes are unchanged OCR, coarse direction, coarse + deskew,
and coarse + deskew + dense refinement. The HTML report scores original-image
quadrilaterals at IoU 0.5. Short-run times are not throughput claims. These remain
synthetic development cases; real sparse/unruled/noisy scans need validation.

## Current results

`pybaseline/results/native_orientation_v1/report.html` contains the 28-page
ablation (27 statement variants plus the size ladder):

| Mode | Exact statement words / 12006 |
|---|---:|
| Correction off | 5377 |
| Coarse page direction | 9321 |
| Coarse + fractional deskew | 9346 |
| Coarse + deskew + dense refinement | 9989 |

The classifier chose the correct quarter turn on all 27 statement cases.
Pure 90/180/270-degree rotations exactly recovered upright OCR counts on each
layout: 277 / 363 / 404, or 287 / 385 / 437 with refinement. Upright and ladder
outputs were preserved. Known +/-0.5 and +/-1.5-degree residuals were estimated
correctly on these ruled fixtures.

Deskew alone improved 9 of 15 fractional cases, worsened 6, and left the other
12 quarter-turn/upright cases unchanged. Correct geometry does not guarantee
better recognition after interpolation. On the 90.5-degree cases it lost
1 / 6 / 4 exact words versus coarse correction alone. Keep deskew separate from
coarse orientation and off by default. These are accuracy trials, not a new
throughput benchmark or a comparison with the complete Python docTR pipeline.

Unit tests cover quarter-turn pixel mapping, fractional affine geometry, a
0.37-degree synthetic skew, blank/single-rule abstention and conflicting directions.
The installed-wheel integration check uses one admission slot and independently
feeds/consumes upright, 90, 180 and 90.5-degree RGB pages:

```powershell
./.venv-baseline/Scripts/python scripts/check_orientation_stream.py
```

It compares text, geometry and original-image quadrilaterals to the native
combined-mode results. The reference path can be changed with `--reference`.
The rebuilt installed wheel passed that check, including a one-slot clean drain
and invalid-option rejection. Confidence comparison preserves its float32
precision because summary JSON widens it to float64. Seventeen Rust tests and
ten Python tests passed; the report and original-coordinate overlay were rendered
and visually checked.
