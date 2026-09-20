"""One versus multiple independent OCR processes with a shared warm-start gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('pybaseline/results/concurrency2'))
    parser.add_argument('--seconds',type=float,default=30)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('workers must be positive')
    if args.output.exists():
        parser.error('Use a new output directory to avoid stale start gates')
    args.output.mkdir(parents=True)
    gate=args.output/'start'
    children=[]; logs=[]
    try:
        for i in range(args.workers):
            out=args.output/f'worker{i}'
            out.mkdir()
            log=(out/'run.log').open('w',encoding='utf-8'); logs.append(log)
            command=[sys.executable,'-m','pybaseline.run','--kind','ocr','--det','fast_base','--reco','parseq',
                     '--output',str(out),'--min-seconds',str(args.seconds),'--gate',str(gate)]
            children.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+600
        while not all((args.output/f'worker{i}/ready').exists() for i in range(args.workers)):
            if any(c.poll() is not None for c in children):
                raise RuntimeError('A worker failed before warmup; inspect run.log')
            if time.monotonic()>deadline:
                raise TimeoutError('Warmup timeout')
            time.sleep(.2)
        gate.touch()
        for child in children:
            if child.wait(timeout=900):
                raise RuntimeError('Worker failed; inspect run.log')
        results=[json.loads((args.output/f'worker{i}/result.json').read_text()) for i in range(args.workers)]
        start=min(r['measured_start_ns'] for r in results)
        end=max(r['measured_end_ns'] for r in results)
        total=sum(r['items_per_trial']*len(r['trials_seconds']) for r in results)
        summary={'workers':args.workers,'pages':total,'wall_seconds':(end-start)/1e9,
                 'aggregate_pages_per_second':total/((end-start)/1e9),
                 'start_spread_seconds':(max(r['measured_start_ns'] for r in results)-start)/1e9,
                 'torch_peak_allocated_sum_bytes':sum(r['memory']['allocated_peak_bytes'] for r in results),
                 'device_peak_bytes':max(r['resources']['device_vram_bytes_max'] or 0 for r in results),
                 'gpu_mean_pct_per_worker':[r['resources']['gpu_util_pct_mean'] for r in results],
                 'note':'Device metrics are shared, never sum GPU utilisation or device VRAM. Makespan includes tail overlap differences.'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate(); child.wait()
        for log in logs:
            log.close()


if __name__=='__main__':
    main()
