"""Record installed docTR merge behavior, including ambiguous/repeated overlaps."""
import json
import random
from pathlib import Path
from doctr.models.recognition.utils import merge_strings

rng = random.Random(29)
cases = []
for _ in range(300):
    a, b = (''.join(rng.choices('abc I_-é', k=rng.randrange(25))) for _ in range(2))
    ratio = rng.choice([0., .1, .25, .5, .75, 1.])
    cases.append([a, b, ratio, merge_strings(a, b, ratio)])
Path('testdata/rust_crop_merge.json').write_text(json.dumps(cases, ensure_ascii=False), encoding='utf-8')
