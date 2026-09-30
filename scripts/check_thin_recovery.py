"""Historical v1 CPU parity; uses the preserved pre-one-hop probe executable."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tune_dense_detection import BASE,read,write,environment
import subprocess

def call(args):
    subprocess.run([str(ROOT/'.cache/native-thin-v1/refinement_probe.exe'),*map(str,args)],env=environment(),check=True)


def main():
    out=ROOT/'pybaseline/results/native_thin_parity';out.mkdir(exist_ok=True)
    write(out/'grid.json',[BASE]);total=0
    for tag,source,expected in [
        ('development','dense_tuning_v2/development','thin_recovery_v3/development'),
        ('validation','dense_tuning_v2/validation','thin_recovery_v3/validation'),
        ('fresh','thin_fresh_v1/baseline','thin_fresh_v1/recovery')]:
        call(['--cache',ROOT/'pybaseline/results'/source/'cache','--grid',out/'grid.json','--output',out/f'{tag}.json','--thin-recovery'])
        native=read(out/f'{tag}.json');prototype={r['page']:r for r in read(ROOT/'pybaseline/results'/expected/'candidates.json')}
        for row in native:
            actual=[w for w in row['words'] if w.get('thin_recovery')]
            expect=prototype[row['page']]['words']
            key=lambda w:tuple(np.array(w['polygon']).ravel())
            actual.sort(key=key);expect.sort(key=key)
            assert len(actual)==len(expect),(row['page'],len(actual),len(expect))
            np.testing.assert_allclose([w['polygon'] for w in actual],[w['polygon'] for w in expect],atol=1e-6,rtol=0)
            total+=len(actual)
    print(f'PASS: 17 pages; {total} native candidates match prototype geometry')


if __name__=='__main__':main()
