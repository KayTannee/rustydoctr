"""Compare native thin proposals against the frozen direct/one-hop prototypes."""
from pathlib import Path
import sys, subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tune_dense_detection import read,write,call,BASE,environment

def main():
    out=ROOT/'pybaseline/results/native_anchor_parity';out.mkdir(exist_ok=True);write(out/'grid.json',[BASE]);total=0
    for tag in ['remaining_60_v1','anchor_holdout_v1']:
        source=ROOT/'pybaseline/results'/tag
        call(['--cache',source/'source/cache','--grid',out/'grid.json','--output',out/f'{tag}.json','--thin-recovery'])
        subprocess.run([str(ROOT/'.cache/native-thin-v1/refinement_probe.exe'),'--cache',str(source/'source/cache'),'--grid',str(out/'grid.json'),'--output',str(out/f'{tag}_old.json'),'--thin-recovery'],env=environment(),check=True)
        old={r['page']:r for r in read(out/f'{tag}_old.json')}
        expected={r['page']:r['words'] for r in read(source/'chain/candidates.json')}
        for row in read(out/f'{tag}.json'):
            actual=[w for w in row['words'] if w.get('thin_recovery')]
            wanted=expected[row['page']]+[w for w in old.get(row['page'],{}).get('words',[]) if w.get('thin_recovery')]
            key=lambda w:tuple(np.asarray(w['polygon']).ravel())
            actual.sort(key=key);wanted.sort(key=key)
            assert len(actual)==len(wanted),(row['page'],len(actual),len(wanted))
            if actual:np.testing.assert_allclose([w['polygon'] for w in actual],[w['polygon'] for w in wanted],atol=1e-6,rtol=0)
            total+=len(actual)
    print(f'PASS: 23 pages; {total} native proposals match frozen prototype geometry')
if __name__=='__main__':main()
