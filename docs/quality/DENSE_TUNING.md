# Dense-tile parameter tuning

Keep the current defaults: the tested alternatives did not improve overall word
accuracy. This experiment changes only refinement tiles; full-page detection
stays fixed at 1536 with its existing parameters.

## Results

Swept 25 combinations of pixel threshold (0.2–0.6) and box expansion
(0.75–1.75), using cached native detector probability maps. Recognized eight
shortlisted configurations. Box score threshold remained 0.1 and morphology
remained 3×3; this is not an exhaustive search of every detector setting.

| Measure | Default: 0.3 / 1.5 | Challenger: 0.2 / 0.75 |
|---|---:|---:|
| Original three pages: correct words / 1,334 | 1,109 | 1,108 |
| Eight validation pages: correct words / 5,960 | 5,471 | 5,369 |
| Validation merged-box candidates | 103 | 164 |

Correct means exact text with a ground-truth box match at IoU ≥0.5. Merge
candidates are geometric flags, not manually confirmed errors. Validation uses
four new layouts/fonts, each upright and skewed 0.6°, rendered at nominal 300 DPI.
The full native pipeline reproduced the probe's text and coordinates on all
11 pages, for both configurations.

Higher thresholds reduced some merges but lost words overall. Tighter boxes
occasionally improved box matching while damaging recognition. The challenger
also varied substantially by font. No default change or throughput claim follows
from these short accuracy runs.

Local results: [HTML report](../../pybaseline/results/dense_tuning_v2/report.html),
[machine-readable recommendation](../../pybaseline/results/dense_tuning_v2/recommendation.json).

## Reproduce

Run VS Code task **OCR: tune dense-tile detection**. It builds the Rust tools and
prepares fixtures/orientation models before running the experiment. Existing OCR
model exports and the baseline Python environment are required. Results go into
a fresh timestamped directory, including fixtures, cached maps, scores and HTML.

Or, after those prerequisites:

```powershell
.venv-baseline/Scripts/python.exe scripts/tune_dense_detection.py
```

## Optional tile-only controls

CLI: `--dense-refine --dense-bin-thresh 0.3 --dense-box-thresh 0.1 --dense-unclip-ratio 1.5`.

Python, before creating `Stream`:

```python
config["dense_refine"] = True
config["dense_detection"] = {
    "bin_thresh": 0.3,
    "box_thresh": 0.1,
    "unclip_ratio": 1.5,
}
```

These are the unchanged defaults. Missing fields retain defaults; invalid values
are rejected. Full-page settings are unaffected. The current priority has moved
to recovering omitted ordinary words, especially thin standalone characters;
word splitting is deferred. See [the missing-word audit](MISSING_WORDS.md).
