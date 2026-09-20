"""Select the upright synthetic corpus for reproducible throughput comparisons."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "testdata/generated"
manifest = json.loads((source / "manifest.json").read_text())
pages = [dict(id=p["id"], image=str(source / (p["id"] + ".png")), words=p["words"])
         for p in manifest["pages"] if p["id"] not in ("a4_skew_7", "a4_mixed_rotation")]
assert all(Path(p["image"]).is_file() for p in pages)
(ROOT / "testdata/throughput.json").write_text(json.dumps({"pages": pages}), encoding="utf-8")
print(f"{len(pages)} pages; {sum(len(p['words']) for p in pages)} truth words")
