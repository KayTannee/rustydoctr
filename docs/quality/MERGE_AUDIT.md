# Merged-word visual audit

Open `output/diagnostics/word-merges/index.html` for six distinct upright layouts,
all 36 page variants, enlarged examples and a before/after dense-refinement switch.
`contact-sheet.png` and `merge-closeups.png` are browser-rendered static sheets.

Red boxes cover at least 50% of the labelled area of two or more reference words.
Blue boxes show those individual words. These are geometric merge candidates;
missing detections and ordinary recognition errors are not highlighted. Original
fixtures use font-metric labels, which are larger than the ink for some characters.
Diagonal axis-aligned boxes can cover multiple words without being a normal
horizontal segmentation failure.

## Results on the 27 statement variants

Three layouts at nine angles, not 27 independent documents. Same 12,006 labelled
words and DB ResNet34 + PARSeq FP32 at detector size 1536.

| Pipeline | Strict exact words | Merged boxes | Containing a 1-2 character token |
|---|---:|---:|---:|
| Page orientation + deskew | 9,346 | 241 | 241 |
| Plus dense refinement | 9,989 | 118 | 118 |

Remaining merges: 60 in deliberate character-stress lines, 27 in slash-separated
reference values, 23 in legal text, eight in diagonal labels. Thus 87/118 (74%)
come from deliberately difficult character/reference lines. Short tokens include
punctuation and digits. This is not an estimate of production-document frequency.

Examples: `I | agree` -> `agree`, `copy. | A` -> `copy.A`, `i | l | 1` -> `111`,
and `Sit | I | a` -> `Sitla`. Some wrong merges have recognition confidence over
99.9%; recognition confidence alone cannot identify them.

Of 2,017 strict failures after refinement, 2,001 lack a one-to-one box match at
IoU 0.5 and 16 have a matched box but wrong text. This is a metric breakdown, not
proof that every failure is caused by detection. At relaxed IoU 0.25 the audit
finds 1,085 words without any overlapping prediction, 429 unmatched words with
merged coverage, 23 other overlaps without a match, and 134 matched wrong words.
Merges explain only part of the remaining loss.

## What the crop experiment actually fixed

747 labelled `I` instances across those 27 variants: 283 correct, 34 read as `-`,
four as `T`, and 426 without a match at IoU 0.25. Guided framing changes these to
285 correct and 32 dashes. The two repairs are the same underlying character on
the 0.5 and 90.5 degree variants, not two independent examples. Strict accuracy
does not improve because the detection boxes remain too short.

No single-character rotation was accepted. Across all 36 pages the policy made
35 upright framing trials and six rotated trials on three words, selecting only
the two upright crops. It uses geometry before recognition, not an explicit
recognized-length <= 2 gate. GPU work is sparse, but the CPU neighbour search
still scans other boxes for each word; add an early size filter and a line/spatial
index before extending this policy.

## Next experiments

1. Inspect raw detector probability, thresholding and morphology on missing thin
   characters. Separate model misses from postprocessing removal and label mismatch.
2. Prototype bounded splitting of boxes containing unusually large internal gaps
   or small neighbouring glyphs. Reuse source pixels and the existing detector map;
   preserve the original alternative until geometry and recognition justify a split.
3. Estimate local direction from a coherent line's longer anchors, then let short
   words inherit it. Restrict recognition retries to short/ambiguous crops; keep
   merge detection separate because merged output can be a long confident string.
4. Validate on new documents before tuning them, then measure sustained overhead.

## Rebuild without inference

```powershell
./.venv-baseline/Scripts/python scripts/merge_contact_sheet.py
```

This regenerates the HTML and JSON evidence from saved predictions. Static PNGs
are screenshots of the contact and close-up sections. The audit stores source
hashes and checks the historical manifest before comparing refinement modes.
