"""Sequential sustained comparison; never overlap GPU benchmark processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True,help='Completed Rust --auto output folder')
    p.add_argument('--output',type=Path,default=Path('pybaseline/results/throughput'))
    p.add_argument('--seconds',type=int,default=300)
    a=p.parse_args();os.chdir(ROOT)
    config=json.loads((a.calibration/'autotune.json').read_text())['selected']
    env=os.environ.copy();runtime=ROOT/'.venv-baseline/Lib/site-packages/onnxruntime/capi'
    env['ORT_DYLIB_PATH']=str(runtime/'onnxruntime.dll')
    env['PATH']=str(runtime)+os.pathsep+str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib')+os.pathsep+env['PATH']
    def run(name,command):
        folder=a.output/name
        print('\nRUN',name,flush=True)
        subprocess.run(command+['--output',str(folder)],check=True,env=env)
        return json.loads((folder/'summary.json').read_text())
    python=[sys.executable,'-u','-m','pybaseline.throughput_python']
    trials=[]
    for batch in [512,1024]:
        r=run(f'tune_ort_d1_r{batch}',python+['--backend','ort','--page-batch','1','--reco-batch',str(batch),'--pages','20'])
        trials.append((r['pages_per_second'],batch))
    best=max(trials)[1]
    (a.output/'python_ort_tuning.json').write_text(json.dumps({'trials':trials,'selected':best},indent=2))
    run('sustained_torch',python+['--backend','torch','--page-batch','4','--reco-batch','1024','--seconds',str(a.seconds)])
    run('sustained_ort',python+['--backend','ort','--page-batch','1','--reco-batch',str(best),'--seconds',str(a.seconds)])
    native=[str(ROOT/'target/release/throughput.exe')]
    for key in ['size','reco_batch','det_batch','workers','arena_mib']:
        native+=['--'+key.replace('_','-'),str(config[key])]
    run('sustained_rust',native+['--inflight',str(config['inflight']),'--seconds',str(a.seconds)])
    run('rust_one_page',native+['--inflight','1','--seconds','120'])
    subprocess.run([sys.executable,'-m','pybaseline.throughput_report','--runs',*[str(a.output/n) for n in ['sustained_torch','sustained_ort','sustained_rust','rust_one_page']],'--output',str(a.output)],check=True)


if __name__=='__main__':main()
