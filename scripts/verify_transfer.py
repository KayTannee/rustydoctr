import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'checksums.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
    path=(root/name).resolve();path.relative_to(root)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('Checksum mismatch: '+name)
print('Verified',len(manifest),'files')
