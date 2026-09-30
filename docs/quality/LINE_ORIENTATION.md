# Experimental line-guided crops

Opt in with `--line-guided-orientation`, or Python config
`config["line_guided_orientation"] = True` after rebuilding the wheel.
It defaults to false; existing profiles preserve their behaviour.

Use after page orientation/deskew, or with inputs already upright. This is a
conservative crop experiment, not a complete mixed-orientation OCR solution.

## Behaviour

- Two nearby horizontal word anchors keep a narrow character upright. An
  alternative crop borrows their vertical extent and adds a small side margin,
  without including a neighbouring detection.
- A tall crop needs two aligned vertical neighbours before trying 90/270 degrees.
  A lone tall character is insufficient evidence. No 45-degree correction yet.
- Original recognition is retained unless a candidate scores at least 0.90,
  improves confidence by more than 0.15, and, for rotation, beats the opposite
  direction by more than 0.10. Rotated single-character candidates cannot win.
- At most 32 extra recognition crops and one million source crop pixels per page.
  They use the existing bounded queues and recognizer; no additional detector pass
  or model. Detection geometry remains unchanged.
- `word.crop_decision` records the evidence, original text/confidence, all tried
  candidates, and selected rotation (null means retained original; 0 means the
  upright context crop won). Confidence is a heuristic, not calibrated certainty.

## Measured outcome

Report: `pybaseline/results/line_orientation_v2/report.html`.
Both modes use DB ResNet34 + PARSeq FP32 at 1536, page orientation, deskew and
dense refinement. Compared 28 existing pages and eight newly generated pages.

| Set | Baseline exact | Guided exact | Extra recognition crops |
|---|---:|---:|---:|
| Existing 28 pages / 12,046 words | 10,003 | 10,003 | 23 |
| New 8 pages / 1,696 words | 1,026 | 1,026 | 18 |

Final strict scoring found zero gains and zero losses. Two upright context crops
were accepted; all ambiguous rotation alternatives were retained as diagnostics
without replacing the original text.

The first trial recovered two `I` characters previously read as dashes at relaxed
IoU 0.25, but **did not improve the strict IoU 0.5 exact-word total**. Their existing
boxes still fail the stricter geometry check. New pages exposed almost-identical
confidence for opposite readings `VI` and `IA`; the final rule abstains on that
ambiguity. The original trial is retained in `line_orientation_v1`.

Those eight new pages became regression fixtures when their failure informed
the abstention rule; they are no longer an untouched holdout. Development labels
use font metrics and new labels use glyph extents: compare modes within each set.
These short accuracy runs do not establish a throughput difference.

Validation: 22 Rust tests, 11 Python tests, and Clippy pass. The installed wheel
matches native text, geometry and crop decisions on five selected pages through
independent RGB producer/consumer threads with one admission slot, including both
repaired characters and ambiguous rotated crops. Confidence comparisons allow
2e-6 FP32 drift when batch composition changes; text and decisions must match.
The Windows comparison readers now explicitly read UTF-8 JSON. Rescoring both
this study and the earlier docTR comparison left the reported totals unchanged.

The larger problem is detection: in the original 28-page refined baseline, 344
ground-truth `I` characters have no overlapping detection, 78 have merged coverage,
and 14 overlap without a one-to-one match at IoU 0.25. Only 38 matched `I` crops
have wrong text. Categories describe geometric evidence, not proven root causes.
Next investigate thin-character suppression and merged-word splitting before
adding more recognition retries. Locally rotated text also needs detection recovery.

## Reproduce

VS Code: **OCR: test line-guided crop orientation** builds Rust, generates samples,
and runs both modes into a fresh timestamped directory.

```powershell
./.venv-baseline/Scripts/python scripts/check_line_orientation.py
# Rebuild an existing report without inference:
./.venv-baseline/Scripts/python scripts/check_line_orientation.py --output pybaseline/results/line_orientation_v2 --report-only
# Installed-wheel RGB producer/consumer parity:
./.venv-baseline/Scripts/python scripts/check_orientation_stream.py --reference pybaseline/results/line_orientation_v2/guided/summary.json
```

`errors.json` contains per-truth-word categories; `changes.json` contains accepted
alternatives; `transitions.json` records strict exact-word gains and losses.
Raw predictions retain rejected alternatives. Provenance hashes cover the binary,
models, manifest and page images. No ground-truth labels guide inference.
