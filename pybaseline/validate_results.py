"""Check result consistency without rerunning GPU work."""
import argparse
import json
from pathlib import Path


def validate(root):
    count=0
    hashes=set()
    for path in root.glob('*/result.json'):
        r=json.loads(path.read_text())
        hashes.add(r['corpus_sha256'])
        assert len(r['trials_seconds'])>=r['config']['repeats'], path
        assert sum(r['trials_seconds'])>=r['config']['min_seconds'], path
        assert all(t>0 for t in r['trials_seconds']), path
        expected=r['items_per_trial']*len(r['trials_seconds'])/sum(r['trials_seconds'])
        assert abs(expected-r['throughput_per_second'])<1e-8, path
        assert len(r['by_page'])==r['pages'], path
        assert r['resources']['samples']>0, path
        assert r['measured_end_ns']>r['measured_start_ns'], path
        predictions=json.loads((path.parent/'predictions.json').read_text())
        if r['config']['kind']=='recognition':
            assert len(predictions)==r['accuracy']['words'], path
            assert sum(p['words'] for p in r['by_page'])==r['accuracy']['words'], path
        else:
            assert len(predictions)==r['pages'], path
            for key in ['truth','predicted','matched','exact']:
                assert sum(p[key] for p in r['by_page'])==r['accuracy'][key], (path,key)
            assert r['accuracy']['matched']<=min(r['accuracy']['truth'],r['accuracy']['predicted']),path
        count+=1
    assert count,'No results'
    assert len(hashes)==1, 'Mixed corpora: do not combine these rankings'
    print(f'Validated {count} results; common corpus hash {next(iter(hashes))}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path,nargs='?',default=Path('pybaseline/results/baseline'))
    validate(p.parse_args().root)
