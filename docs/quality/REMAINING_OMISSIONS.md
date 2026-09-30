# Remaining omissions: audit and one-hop context experiment

Production code/settings are unchanged. The new context rule is a diagnostic
prototype, evaluated against existing native thin recovery.

## Where the remaining 60 words went

| Furthest stage reached | Words |
|---|---:|
| Thin component found, fewer than two direct line anchors | 27 |
| No pixels above the current threshold in the labelled window | 12 |
| Component too small or fragmented | 9 |
| Shape rejected | 6 |
| Expanded recovery crop overlaps an existing detection | 5 |
| Both dense tiles reject the word at their ownership boundary | 1 |

No cases in this subset reached a recovery proposal and then disappeared solely
at the recognition gate. Labels locate diagnostic windows only; recovery never
uses expected text or ground-truth coordinates. Categories select the furthest
component stage across eligible full-page and tile maps. Pixel-window evidence
can include partial components, so these are pipeline diagnostics, not claims
about the model's internal reasoning.

[All 60 highlighted with stage evidence](../../pybaseline/results/remaining_60_v1/report.html)
and [PNG contact sheet](../../pybaseline/results/remaining_60_v1/contact-sheet.png).

## One targeted experiment

For a discarded thin component with **exactly one** direct text anchor, allow
that anchor to supply one more aligned word farther along the same line. The
second word must lie beyond the first, with compatible height/vertical position
and bounded gap. No recursive chains, language rules or lower pixel thresholds.
Existing recognition acceptance remains ≥0.9 confidence and 1–2 alphanumeric
characters. Existing words/boxes are preserved.

| Set | Proposed / accepted additions | Correct text with ≥50% reference coverage | Ordinary no-overlap omissions |
|---|---:|---:|---:|
| Original 17 page variants | 24 / 22 | 21, plus one insufficient box | 60 → 38 |
| Fresh six variants | 16 / 16 | 16 | 20 → 4 |

Every accepted transcription was `I`. The two rejected original-set candidates
read as dashes. All 22 new overlaps on the original set belong to the 27 anchor
failures, reducing that group to five. One added I box covers too little of its
label; it remains a geometry limitation and is excluded from the 21 validated
additions. Strict IoU 0.5 exact-word gain was eight on the original set and zero
on the fresh set: most new boxes remain loose despite recovering readable text.

Fresh fixtures were generated after freezing the rule: Calibri/Verdana, slight
blur, upright / −0.8° / 180°, prose, numeric fields, table rules, barcode-like
bars, and white text on black. All 16 fresh additions matched labelled words;
none appeared on the negative-control marks. These are two new layouts repeated
at three angles, not six independent documents. No real-document validation or
throughput claim is made for this new rule.

The prototype caps additional proposals at 32 and additional crop area at one
million source pixels per page. Existing native recovery is also present in the
baseline. Combined old/new proposals peaked at ten per original page and six
per fresh page; a native port should share the existing overall recovery budget.

[Before/after table and every accepted crop](../../output/diagnostics/anchor-chain/index.html).

## Confirmed tile-boundary defect

`agree` on `statement_0_cw0` is detected in both tiles. The ownership boundary is
x=1240 source pixels. The left tile's detection centre is x=1242.24; the right
tile's is x=1239.14. Both are outside their respective ownership interval and
both are discarded. This is a reconciliation defect, not weak model confidence.
The saved audit includes both rejected boxes and their ownership intervals.
It is identified but not fixed in this experiment.

## Reproduce and next implementation

With the saved maps and baseline environment:

```powershell
.venv-baseline/Scripts/python.exe scripts/audit_remaining_omissions.py
.venv-baseline/Scripts/python.exe scripts/experiment_anchor_chain.py --source pybaseline/results/remaining_60_v1/source --output pybaseline/results/my_anchor_trial
.venv-baseline/Scripts/python.exe scripts/prepare_anchor_validation.py --output pybaseline/results/my_anchor_holdout
.venv-baseline/Scripts/python.exe scripts/experiment_anchor_chain.py --source pybaseline/results/my_anchor_holdout/source --output pybaseline/results/my_anchor_holdout/chain
.venv-baseline/Scripts/python.exe scripts/report_anchor_chain.py pybaseline/results/my_anchor_trial pybaseline/results/my_anchor_holdout/chain
```

Use fresh output directories. The audit is CPU-only; recognition and fresh-map
generation use the GPU. Script snapshots and source/model hashes are saved with
the experiments. The native cache probe gained `--dump-tiles` solely for tracing
pre-ownership detections. Eighteen Python tests and probe Clippy pass; both HTML
reports were checked at desktop/mobile widths.

Recommended next work: fix cross-tile ownership without losing detections or
duplicating words, then port the validated one-hop rule into opt-in native
recovery with the same shared budgets and repeat the native quality check.


## Native implementation follow-up

Both changes are now implemented. One-hop recovery is behind the existing
`--thin-recovery` / Python `config["thin_recovery"] = True` opt-in. Seam
reconciliation applies when dense refinement is enabled. The preceding audit
and prototype results describe the preserved v1 behavior; their reproduction
uses `.cache/native-thin-v1/refinement_probe.exe`.

Full native old/new inference on 23 variants added 39 words, removed none and
changed no existing text/box. Of those, 38 meet exact-text plus 50% labelled-word
coverage; one `I` remains a partial-box case. The original 60 omissions become
37; the holdout 20 become 4. The recovered seam word is `agree`. All 52 candidate
geometries match the saved direct/one-hop prototypes before recognition.

```powershell
# Requires the saved 23-page manifests and preserved v1 executable.
.venv-baseline/Scripts/python.exe scripts/check_anchor_recovery.py
.venv-baseline/Scripts/python.exe scripts/benchmark_anchor_recovery.py --output pybaseline/results/my_native_anchor --old-binary .cache/native-thin-v1/throughput.exe --pages 230
```

The benchmark runs quality old/new, then 230 pages per trial in old/new and
new/old order. Both versions enable thin recovery. It excludes model loading
and a full-workload warmup, retains a three-page admission limit, and records
binary hashes and device/idle-relative VRAM. Existing output resumes completed
runs; use a new folder for a new implementation. This is not a 4GB-budget test.

[Native report and added-word crops](../../pybaseline/results/native_anchor_v1/report.html).
