"""Run frozen public test splits through stock docTR and native Rust; resume by summary."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tune_dense_detection import environment,read,write
MODES=['python_1024','rust_1024','python_1536','rust_1536','python_rotated_1536','rust_enhanced_1536']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'pybaseline/results/public_v1');p.add_argument('--datasets',nargs='+',default=['funsd','cord-v2','hiertext']);p.add_argument('--modes',nargs='+',default=MODES,choices=MODES);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for dataset in a.datasets:
        prepared=ROOT/'testdata/public'/dataset/'prepared';pages=read(prepared/'manifest.json')
        for mode in a.modes:
            out=a.output/dataset/mode
            if (out/'summary.json').exists():
                assert read(out/'summary.json')['pages']==len(pages);continue
            print('START',dataset,mode,len(pages),'pages',flush=True)
            size='1024' if mode.endswith('1024') else '1536'
            if mode.startswith('python'):
                cmd=[sys.executable,str(ROOT/'pybaseline/public_doctr.py'),'--manifest',str(prepared/'manifest.json'),'--output',str(out),'--size',size]
                if 'rotated' in mode:cmd+=['--rotate']
            else:
                cmd=[str(ROOT/'target/release/throughput.exe'),'--workload',str(prepared/'workload.json'),'--output',str(out),'--pages',str(len(pages)),'--size',size,'--reco-batch','256','--det-batch','1','--workers','2','--inflight','3','--arena-mib','6144','--det-arena-mib','4096']
                if 'enhanced' in mode:cmd+=['--page-orientation','--deskew','--dense-refine','--thin-recovery']
            subprocess.run(cmd,cwd=ROOT,env=environment(),check=True)
            assert read(out/'summary.json')['pages']==len(pages)
            write(out/'inputs.json',dict(manifest_sha256=hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((ROOT/'target/release/throughput.exe').read_bytes()).hexdigest() if mode.startswith('rust') else None,command=cmd))
    print('All requested inference runs complete.',flush=True)
if __name__=='__main__':main()
