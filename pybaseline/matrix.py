"""Sequential isolated model sweep; resume completed cases and record failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

DETECTORS = ['fast_tiny', 'fast_small', 'fast_base', 'db_mobilenet_v3_large', 'db_resnet34', 'db_resnet50', 'linknet_resnet18', 'linknet_resnet34', 'linknet_resnet50']
RECOGNIZERS = ['crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'crnn_vgg16_bn', 'vitstr_small', 'vitstr_base', 'parseq', 'viptr_tiny', 'sar_resnet31', 'master']


def cases():
    for arch in DETECTORS:
        yield 'det_'+arch, ['--kind','detection','--det',arch]
    for arch in RECOGNIZERS:
        yield 'reco_'+arch, ['--kind','recognition','--reco',arch]
    for det, reco in [('fast_base','parseq'), ('fast_base','crnn_vgg16_bn'), ('db_resnet50','parseq'), ('db_mobilenet_v3_large','crnn_mobilenet_v3_small')]:
        yield f'ocr_{det}_{reco}', ['--kind','ocr','--det',det,'--reco',reco]
    for name, flags in [('rotation_crops',['--rotation','crops']), ('rotation_straighten',['--rotation','straighten']),
                        ('size1536',['--size','1536']), ('stretch',['--stretch']),
                        ('batch4',['--batch','4','--reco-batch','256']), ('threads1',['--threads','1'])]:
        yield 'ocr_fast_base_parseq_'+name, ['--kind','ocr','--det','fast_base','--reco','parseq']+flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('pybaseline/results/baseline'))
    p.add_argument('--min-seconds', type=float, default=20)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--only', default='', help='substring filter')
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name, flags in cases():
        if a.only not in name:
            continue
        out = a.output/name
        if (out/'result.json').exists():
            print('SKIP', name, flush=True); continue
        out.mkdir(exist_ok=True)
        command = [sys.executable,'-m','pybaseline.run',*flags,'--output',str(out),'--min-seconds',str(a.min_seconds),'--repeats',str(a.repeats)]
        print('START',name,flush=True)
        start = time.time()
        with (out/'run.log').open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                status = {'returncode': result.returncode}
            except subprocess.TimeoutExpired:
                status = {'error': 'timeout', 'timeout_seconds': a.timeout}
        status.update(command=command, elapsed_seconds=time.time()-start)
        (out/'status.json').write_text(json.dumps(status,indent=2), encoding='utf-8')
        print('END',name,status,flush=True)


if __name__ == '__main__':
    main()
