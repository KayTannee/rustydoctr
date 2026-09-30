"""Verify ordered drain and repeated text/geometry stability in completed native trials."""
import argparse
import json
from pathlib import Path
import numpy as np


def signature(record):
    # Summary Value serialization widens f32; JSONL serializes the original f32.
    return [(w['text'],np.asarray(w.get('quadrilateral',w['polygon']),dtype=np.float32).tolist(),bool(w.get('thin_recovery'))) for w in record['words']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args();out=args.results
    runs=json.loads((out/'runs.json').read_text(encoding='utf-8'));reports=[]
    for run in runs:
        if run['folder'].startswith('quality'):continue
        summary=json.loads((out/run['folder']/'summary.json').read_text(encoding='utf-8'))
        expected={r['id']:signature(r) for r in summary['first_pages'].values()}
        changed=[];count=0;recovered=0
        with (out/run['folder']/'pages.jsonl').open(encoding='utf-8') as stream:
            for record in map(json.loads,stream):
                assert record['sequence']==count,(run['folder'],'sequence gap',count)
                if signature(record)!=expected[record['id']]:changed.append(dict(sequence=count,id=record['id']))
                recovered+=sum(bool(w.get('thin_recovery')) for w in record['words'])
                count+=1
        assert count==run['pages'],(run['folder'],'drain count mismatch')
        reports.append(dict(run=run['folder'],pages=count,changed_pages=len(changed),examples=changed[:10],recovered_words=recovered))
    (out/'repeat_checks.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))
    assert all(r['changed_pages']==0 for r in reports),'Repeated output changed; inspect repeat_checks.json'


if __name__=='__main__':main()
