# Quality work: review checkpoint

Status: initial accuracy experiments complete. The GPU was subsequently freed;
an opt-in Rust dense-refinement path and isolated throughput comparison were
authorized. Defaults remain unchanged. See QUALITY_FINDINGS.md for initial
experiments and DENSE_REFINEMENT.md for the native implementation.

## Review samples

Run VS Code task **OCR: generate quality sample pages (CPU only)**, or:

```powershell
./.venv-baseline/Scripts/python scripts/generate_document_fixtures.py
```

Outputs under `output/pdf/quality/`:

- `statements.pdf`: three A4 statement designs with large inverse titles,
  label/value pairs, addresses, real Code 128 barcodes, ruled/striped tables,
  and 7 / 6 / 5.5 pt legal text. Standalone I/a and punctuation appear in several
  regions. Side references are rotated 90 degrees; an Amount / I label is at 45.
- `text_scale.pdf`: the same `Sit I a` text at 144, 96, 72, 48, 36, 24, 18, 12,
  10, 8 and 6 pt. This separates word size from word content.
- 28 labelled PNG cases: each statement at 0, ±0.5, ±1.5, 90, 180, 270 and
  90.5 degrees, plus the size ladder. Rendering is 300 DPI.
- `manifest.json`: exact drawing-derived word polygons, source font sizes,
  semantic regions, local text angles, whole-page angles and affine transforms.
  Barcodes are labelled non-text regions. Quarter-turn images use exact pixel
  permutations; only non-right-angle skews introduce interpolation.

Labels use font metrics rather than tight ink contours. Detection matching will
therefore be reported at more than one IoU threshold. Blank barcode spaces and
table rules must not count as missing text.

## 1. Establish scale behaviour before choosing a policy

Keep the model pair and FP32 weights fixed. Test detector inputs 1024, 1536, 2048
and 2560 using the same 300-DPI pixels. Report each font size and document region
separately, including misses, merges, splits and exact text. Do not infer quality
from aggregate recognition confidence.

Then repeat selected cases with different source render DPIs at a fixed detector
input size. A larger source raster still becomes the same detector tensor unless
we explicitly change its dimensions or use tiles. These are separate experiments.

The size ladder directly tests whether enlarged letters stop being detected or
recognized. Loss of complete large-word detections was observed in the ladder;
this does not establish a universal limit. Test full-width bands alongside resizing, with an
unchanged effective text scale where possible.

`pybaseline/quality_study.py` is an exploratory ONNX accuracy experiment harness.
It requires an explicit `--run-inference` opt-in. It is not a production API or a
completed benchmark. Its initial timings include first-shape warmup and cannot
be used for throughput claims. Add warmed repetitions before comparing cost.

## 2. Find suspicious regions without ground-truth knowledge

A confident recognized word can still cover two merged words, and a missed word
has no recognition confidence. Candidate evidence should combine:

| Signal available during inference | What it can suggest | Main false alarm |
|---|---|---|
| Small text-like image components at detector scale | Insufficient sampling | Punctuation, noise, barcode bars |
| Text-like ink not covered by detector boxes | Missed regions | Rules, logos, graphics |
| An unusually wide box with several internal gaps | Merged words | Wide words, serial numbers |
| Repeated tiny or fragmented detector components | Fragmented segmentation | Isolated punctuation |
| Weak detector boundaries or uncertain probability-map edges | Poor separation | Antialiasing, low contrast |
| Low recognition confidence, considered with geometry | Poor crop/orientation | Unusual but valid text |

Use density and line/region agreement, not a single threshold. Exclude long rules
and barcode-like repeated bars from direction votes and text-density triggers.
Never silently discard nearby address text just because it borders a barcode.

The first experiment compares fixed full-page resolution, fixed overlapping
bands, and selectively refined regions. Log every trigger, crop and decision so
we can see why something was rescanned. Use one statement variant to tune rules,
another to validate and the third as a held-out case; rotations of one layout
must stay in the same split. All three current layouts were inspected during
development, so fresh untouched layouts are now needed for that validation.
Synthetic results will still need real-document validation later.

## 3. Separate page direction, fine skew and local text direction

**Page direction:** distinguish 0/90/180/270 using a small page-orientation model.
Geometry alone cannot reliably distinguish upright from upside down. The docTR
orientation weights are already cached locally; test its confidence and failure
cases before relying on it. Evaluate running this small model on CPU to preserve
the GPU memory budget.

**Fine skew:** estimate a floating-point residual from multiple long text lines
and/or reliable table rules. Retain the float through correction and output.
Use consensus and report insufficient evidence rather than pretending every page
has a reliable angle. Sparse pages and locally rotated content need dedicated
tests. In the installed docTR 1.1.0 source, `models/_utils.py` explicitly rounds
the median angle inside `estimate_orientation` (line 129); its current return
annotation is `int`. The fractional-angle loss is real in this local version.

**Local direction:** after global correction, upright is a prior, not a rule.
Use neighbouring words and line geometry to identify coherent rotated groups.
An isolated I, dash or underscore is weak orientation evidence. Such tokens
should inherit a well-supported line direction instead of being independently
flipped. A genuine vertical reference or diagonal table label can override the
page prior when its own group has evidence. Do not replace a recognized character
with I merely because that would make a sentence read better.

Where ambiguity remains, expose it and cap retries. Compare alternate
recognitions of the affected crop/group rather than all words on the page.

## 4. Reuse work inside the native pipeline

The proposed sequence is:

1. CPU page-direction/fine-skew analysis; rectify the source once if justified.
2. One normal detector pass; retain its probability map and initial geometry.
3. Select a bounded number of suspicious regions, using source pixels and maps.
4. Detect only those higher-detail or locally rectified regions.
5. Reconcile overlap and box replacements **before recognition**, then recognize
   the final crops in the existing cross-page batches.
6. Retry recognition only for remaining orientation ambiguities. Reuse accepted
   text/crops; do not rerun the full document to combine alternate settings.

A rotated probability map is not equivalent to a fresh network prediction;
reuse it as evidence, not as an unvalidated substitute for inference. Store
affine transforms so every returned word maps back to the original page.
Rotated words need quadrilateral coordinates; decide a backward-compatible
response extension before changing the current two-corner `polygon` field.

Keep the existing admission credits and bounded queues. A page retains its
credit until its refinements finish. Cap refinement area, region count, scale,
crop count and orientation retries; tune those limits from measurements rather
than guessing defaults now. Continue using one detector and one recognizer
session, with no model-per-worker duplication. Extra GPU passes have a real
cost; CPU overlap cannot make them free.

## Acceptance evidence before enabling a default

- Region-wise detection recall, merge/split counts and exact word recall.
- I/a/-/_/em-dash confusion counts, including table cells and legal text.
- Fractional skew error on ±0.5-degree pages; coarse orientation accuracy.
- Recovery of local 45/90-degree text without flipping upright single characters.
- Barcode/rule false positives, duplicate words and tile-boundary truncations.
- Mapping correctness for original-page quadrilaterals and expanded rotations.
- Warm throughput, latency, detector pixels/passes, recognition retries, peak
  incremental VRAM and queue limits, using the same machine profile as before.

First ship an opt-in quality mode, preserving the measured upright fast path.
Promote defaults only after the held-out cases show an acceptable quality/cost
trade-off. Next step is review of the accuracy results before promoting an experimental policy.
