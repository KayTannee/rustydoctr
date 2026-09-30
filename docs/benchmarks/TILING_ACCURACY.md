# Resolution and tile overlap experiment

Run the frozen development comparison:

```powershell
.venv-baseline/Scripts/python.exe scripts/prepare_tiling_development.py
.venv-baseline/Scripts/python.exe scripts/benchmark_tiling.py --arena-mib 8192
.venv-baseline/Scripts/python.exe scripts/remerge_tiling.py
.venv-baseline/Scripts/python.exe scripts/report_tiling.py
```

Results: `pybaseline/results/tiling_dev_v1/report.html`. Completed modes resume
from their summary; use a fresh `--root` for changed experiments. Preparation
currently writes the default root. The installed Rust wheel and models must exist.

## Frozen protocol

- 50 FUNSD training pages and 200 HierText validation images selected by SHA-256
  of image ID; all 100 CORD v2 validation pages. No selection using OCR outcomes.
- Same native DB ResNet34 / PARSeq models; rotation, deskew, character recovery
  and line alternatives disabled for every mode. Original image pixels retained.
- Plain whole-page detection at 1024 and 1536.
- Whole-page 1024 plus existing native selective dense-band refinement.
  This is the current image-based small-text selector, not a new confidence gate.
- Full 1024 detector tiles at effective page long side 1536, overlap 128.
- Full 1024 detector tiles at effective page long side 2048, overlaps 0, 128, 256.
- Repeat 2048 / overlap 128 with the grid shifted half a stride on both axes.

Effective resolution matters: at 2048, each square source crop spans half the
original page's longest side, then resizes to detector input 1024. At 1536 it
spans two thirds. Upsampling an existing scan does not add source detail.
Grid stride is tile side minus overlap; shift can add edge tiles. White padding
maintains the same scale in partial page-edge tiles. No page region is omitted.

Full tiling is an accuracy prototype around the Rust Python library, not a new
production tiler. It recognizes every tile and merges predictions afterwards:
text-independent IoU >=0.5 suppression, prioritising distance from interior crop
edges then detector objectness. Raw observations remain in `raw_tiles.jsonl`.
The native selective path instead reconciles detections before recognition and
uses its existing seam fix. Their merge policies differ and are recorded openly.

## Evaluation

Uses the same public scorers as `PUBLIC_BENCHMARKS.md`. Reports detection and
exact end-to-end F1, short-word diagnostics, and paired gained/lost labels.
Supplemental polygon assignment diagnostics count boundary-crossing labels,
labels with no complete view in any tile, missed words and duplicate predictions.
Boundary examples are associations, not causal proof; shifting a crop also changes
its surrounding context. Native selective ownership seams are counted separately.

Cold elapsed times include model loading, Python merging and redundant per-tile
recognition. They show experimental cost, not optimised production throughput.
No defaults or production OCR algorithms change in this experiment.

The optional `_guarded` rows reuse exactly the same inference, adding an edge
rule before NMS: reject predictions within 8 detector pixels of an interior edge
only if another tile provides at least 16 pixels of context around that box.
This can discard fragments, but also loses a word if the alternative view misses
it; losses remain in the scores. The 8/16 constants are fixed for this experiment,
not chosen by sweeping labelled outcomes. Guarded rows have no independent timing.


## Heatmap fusion

Build the experimental native probe, then run the same four overlapping grids:

```powershell
$env:CARGO_HOME = "$PWD/.cache/cargo"
& "$env:USERPROFILE/.cargo/bin/cargo.exe" build --release --locked --bin heatmap_probe
.venv-baseline/Scripts/python.exe scripts/benchmark_heatmap.py
.venv-baseline/Scripts/python.exe scripts/report_tiling.py
.venv-baseline/Scripts/python.exe scripts/report_size_strata.py
.venv-baseline/Scripts/python.exe scripts/heatmap_contact_sheet.py
.venv-baseline/Scripts/python.exe scripts/validate_tiling_results.py
```

`heatmap_uniform` averages sigmoid probabilities in overlap regions.
`heatmap_weighted` weights tile centres more strongly, tapering linearly over
128 detector pixels at interior edges (minimum weight 0.001). Accumulated weights
are normalized per pixel. Tile predictions are aligned with bilinear sampling
onto an unpadded rectangular map at the effective page resolution. DB thresholds,
morphology and expansion run once on that map. The recognizer then reads original
page crops with normal wide-word splitting; no rotation or recovery is added.

Both policies share the same detector passes. The first page's pre-fusion tile
boxes/objectness are checked against the earlier library outputs. Selected map
PNGs, controls and hashes are saved. Only one page's maps are resident at a time;
this serial probe is for accuracy, not a finished streaming implementation.


One HierText shifted-grid box run exhausted the original 2 GiB recognizer arena.
Its partial output is retained under `failed_attempts/`. That mode is rerun with
`--arena-mib 8192` (4 GiB detector + 4 GiB recognizer), retaining batch 256 and the
same OCR rules. This changes the memory budget, not the accuracy configuration.


## Size-stratified diagnosis

After generating the report, run `scripts/report_size_strata.py` to create
`size_strata.html`. Groups use labelled vertical extent at hypothetical whole-page
1024 input, fixed across configurations. This is a proxy for text size, not font
size; rotated/vertical labels can have misleadingly large extents. Values are
strict polygon detection/exact recall, not precision or F1.

Initial development evidence: HierText's 6,284 labels below 8 pixels at 1024 improve
from 14.3% to 33.9% exact recall at 1536. FUNSD's 1,707 labels in the 8–12 band
improve from 67.2% to 71.1%, while its 20–40 band falls from 65.7% to 57.9%.
CORD has only five of 2,186 labels below 12 pixels in this sample. This supports
investigating size-aware region selection; it does not establish an optimal
text-size threshold or prove that size alone causes all errors.

HierText heatmap inputs are decoded with Pillow and saved losslessly as PNG so
both comparison paths consume identical RGB pixels. A preliminary run decoded
JPEGs directly in Rust and failed the detector-box consistency check; it is kept
under `failed_attempts/` and excluded from scoring. The corrected runs retain
strict first-page per-tile coordinate/objectness checks.


## Recommended next work

1. Keep these broad corpora as regression checks. Add a focused mixed-size
   document benchmark with large headings, ordinary body text and genuinely small
   footnotes, separating high-detail rendered PDFs from already-blurred scans.
2. Use size-stratified recall plus false positives and losses of previously correct
   words. Whole-corpus F1 alone obscures benefits concentrated in small-text bands.
3. Establish the remaining headroom: evaluate recognition on labelled word crops,
   and a diagnostic rescan of labelled tiny-text regions. Label-guided selection is
   an upper-bound experiment, never a deployable selector or published test claim.
4. If that headroom is useful, improve automatic region selection using estimated
   text scale, density and unexplained text-like content. Recognition confidence
   alone misses absent words and can be high on incorrect text. Avoid blanket
   upscaling of low-detail receipts or selecting table rules/noise as small text.
5. Use local overlapping heatmap fusion as a candidate for selected regions,
   preserving original detections elsewhere. Require measured small-word gains,
   bounded false positives and minimal loss of already-correct words before
   enabling it. Then measure cost in the bounded streaming pipeline.

Do not treat the synthetic thin-character gains as general recovery evidence;
public tests showed very few label-verified additions. Keep those heuristics
opt-in while diagnosing failure causes, rather than layering on language guesses.


The 2048/128 weighted heatmap run raises HierText's under-8-pixel group to 43.2%
exact recall, versus 14.3% for plain 1024; zero-overlap omissions fall from 2,852
to 1,000. Its aggregate F1 is nevertheless below 1536 fusion. Tiny-word gains
and whole-page regression can coexist; this is the motivation for region gating,
not a reason to replace every page with higher-resolution tiling.


Source scale also limits the relevance of blanket upscaling: every FUNSD image
in this development sample has a 1,000-pixel long side, so the 1024 baseline
already retains its source resolution. These differ from a 300-DPI vector PDF
page whose small print contains real detail that a whole-page 1024 pass discards.
CORD's median source long side is 1,296 pixels, but its labelled text is mostly
already large at detector scale. HierText's median source long side is 1,600.
