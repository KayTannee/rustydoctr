# Missing ordinary words

Current priority: recover omitted words in prose and fields, especially standalone
`I`, `a` and digits. Artificial character/slash sequences are secondary; word
splitting and language correction are deferred.

Run the CPU-only saved-output audit:

```powershell
.venv-baseline/Scripts/python.exe scripts/audit_missing_words.py
```

Open [the visual report](../../output/diagnostics/missing-words/index.html). Inputs are
the saved line-orientation comparison and dense-tuning-v2 results. No new GPU
inference is needed. Explicit known stress sequences are separated using fixture
text, never inferred from OCR errors. Ordinary sentences preceding those
sequences remain in the primary group. Locally rotated reference text and pure
punctuation are reported separately, not removed from the evidence.

## Findings

Across 27 original statement variants and eight new dense-page variants:

- Ordinary words: 15,497 labels; 320 have no overlapping predicted box.
- 316 of those 320 are `I`. Lowercase `a` has zero complete omissions.
- These counts repeat content across rotations: seven independent source layouts.
- Digit coverage is insufficient: only 1, 2 and 3 occur outside the artificial
  strips, and there are few distinct examples.

Missing IoU matches are not automatically omissions. The audit separately counts
correct text with loose boxes, wrong recognised text, merged coverage, partial
coverage and no overlap. The last category is visual evidence to investigate,
not a guarantee that a human would label every case identically.

For 26 omitted ordinary words inside cached dense-tile ownership, the labelled
word windows contained above-threshold pixels in 25 cases. In 24 cases those
pixels disappeared during 3×3 morphological opening; one survived, and one was
below threshold initially. This traces an important failure stage but does not
yet demonstrate that a recovery rule improves end-to-end accuracy. Full-page
maps were not cached, so these 26 cases cannot explain all 320 omissions.

## Next experiment

Completed the first bounded recovery experiment: **9 ordinary I omissions
recovered across 17 page variants**, with no accepted false positives in that
sample. The rule is now available as opt-in native recovery; see
[THIN_RECOVERY.md](THIN_RECOVERY.md) for the subset, limits and reproduction commands.

Compare the raw threshold mask with the cleaned mask. Consider only narrow,
discarded components near an established text line, with compatible height and
baseline; retain bounded candidate and crop budgets. Test recognition before
accepting recovered words, measure false positives on rules/barcodes/punctuation,
and add ordinary numeric-field examples spanning digits 0–9. Avoid globally
disabling cleanup. Leave existing boxes and splitting policy unchanged initially.

The previous threshold/expansion sweep remains documented in
[DENSE_TUNING.md](DENSE_TUNING.md); it did not justify changing defaults. Its
aggregate word score includes stress sequences and is not the new primary metric.
