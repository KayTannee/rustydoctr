"""Check completed development experiment artifacts and retain final provenance."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'pybaseline/results/tiling_dev_v1';checks=[]
    for ds in ['funsd','cord-v2','hiertext']:
        pages=json.loads((root/ds/'manifest.json').read_text(encoding='utf-8'));expected=[p['id'] for p in pages]
        files=list((root/ds).glob('**/scores.json'));assert len(files)==20,(ds,len(files))
        for file in files:
            score=json.loads(file.read_text(encoding='utf-8'));assert score['pages']==len(pages)
            ids=[]
            for i,line in enumerate((file.parent/'pages.jsonl').open(encoding='utf-8')):
                row=json.loads(line);assert row['sequence']==i;ids.append(row['id'])
            assert ids==expected,str(file);checks.append(dict(run=str(file.parent.relative_to(root)),pages=len(ids)))
        for control in (root/ds).glob('*_heatmap/controls.json'):
            values=json.loads(control.read_text(encoding='utf-8'));assert values and all(c['objectness_max_difference']<1e-5 for c in values)
    files=[ROOT/'Cargo.toml',*ROOT.glob('scripts/*tiling*.py'),*ROOT.glob('scripts/*heatmap*'),ROOT/'scripts/report_size_strata.py',ROOT/'pytests/test_tiling.py',ROOT/'target/release/heatmap_probe.exe']
    result=dict(completed_score_sets=checks,tooling_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},note='Final hashes; per-run provenance and failed attempts remain separately recorded.')
    (root/'final_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('Validated 60 score sets, ordered complete page outputs, and heatmap detector controls.')
if __name__=='__main__':main()
