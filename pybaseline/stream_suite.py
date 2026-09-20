"""Tune tested batch sizes, then run three five-minute streaming workloads."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,default=Path('pybaseline/results/rotation_screen_v2/selection.json'))
    p.add_argument('--output',type=Path,default=Path('pybaseline/results/streaming'))
    p.add_argument('--seconds',type=float,default=300)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    def run(name,page,batch,reco_batch,seconds):
        folder=a.output/name
        if (folder/'summary.json').exists():
            return json.loads((folder/'summary.json').read_text())
        if folder.exists():
            raise RuntimeError(f'Incomplete run at {folder}; choose a fresh suite output directory')
        cmd=[sys.executable,'-m','pybaseline.stream','--selection',str(a.selection),'--page',page,
             '--batch',str(batch),'--reco-batch',str(reco_batch),'--seconds',str(seconds),'--output',str(folder)]
        print('START',name,flush=True)
        with (a.output/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=seconds+600)
        if result.returncode:
            print('FAILED',name,'see log',flush=True)
            return None
        result=json.loads((folder/'summary.json').read_text())
        print('DONE',name,result['pages_per_second'],'pages/s',flush=True)
        return result
    candidates=[]
    for batch,reco in [(1,128),(4,256),(8,512),(16,1024)]:
        result=run(f'tune_b{batch}_r{reco}','upright_characters',batch,reco,20)
        if result:
            candidates.append(result)
    if not candidates:
        raise RuntimeError('All batch configurations failed')
    # A configuration is eligible only if batching preserves the warmup sample accuracy.
    max_exact=max(c['accuracy_sample']['exact'] for c in candidates)
    candidates=[c for c in candidates if c['accuracy_sample']['exact']==max_exact]
    best=max(candidates,key=lambda r:r['pages_per_second'])
    batch=best['config']['batch']; reco=best['config']['reco_batch']
    selection={'batch':batch,'reco_batch':reco,'selection':str(a.selection),
               'reason':'Highest full-pipeline throughput among tested batches with the highest unchanged warmup-sample exact count. Short tuning trial, not exhaustive optimization.'}
    (a.output/'batch_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    for page in ['upright_characters','skew7_characters','skew7_mixed_characters']:
        if run('long_'+page,page,batch,reco,a.seconds) is None:
            raise RuntimeError(f'Long run failed: {page}')


if __name__=='__main__':
    main()
