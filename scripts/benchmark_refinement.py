"""Compare warmed native upright OCR, dense refinement and full-page 2560."""
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pybaseline.metrics import score_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=int, default=96)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.pages < 3 or args.pages % 3:
        parser.error('--pages must be a positive multiple of three')
    out = args.output or ROOT/'pybaseline/results'/('native_refinement_'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    fixtures = ROOT/'output/pdf/quality'
    manifest = json.loads((fixtures/'manifest.json').read_text())
    pages = [p for p in manifest['pages'] if p['id'] in ['statement_0_cw0','statement_1_cw0','statement_2_cw0']]
    assert len(pages) == 3
    workload = out/'workload.json'
    workload.write_text(json.dumps({'pages':[{'id':p['id'],'image':str((fixtures/p['image']).resolve())} for p in pages]}))
    runtime = ROOT/'.venv-baseline/Lib/site-packages/onnxruntime/capi'
    env = os.environ.copy()
    env['ORT_DYLIB_PATH'] = str(runtime/'onnxruntime.dll')
    env['PATH'] = str(runtime)+os.pathsep+str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib')+os.pathsep+env['PATH']
    files = [ROOT/'target/release/throughput.exe', ROOT/'models/db_resnet34.onnx', ROOT/'models/parseq.onnx', fixtures/'manifest.json']
    (out/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}, indent=2))
    rows = []
    for repetition, modes in enumerate([['base','refined','full2560'],['full2560','refined','base']],1):
        for mode in modes:
            folder = out/f'{repetition}_{mode}'
            cmd = [str(ROOT/'target/release/throughput.exe'),'--workload',str(workload.resolve()),'--output',str(folder.resolve()),
                   '--pages',str(args.pages),'--size','2560' if mode=='full2560' else '1536',
                   '--reco-batch','256','--det-batch','1','--workers','2','--inflight','3',
                   '--arena-mib','6144','--det-arena-mib','4096']
            if mode=='refined':cmd.append('--dense-refine')
            print(f'\nTrial {repetition}: {mode}',flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,check=True,timeout=1200)
            summary=json.loads((folder/'summary.json').read_text())
            scores=[score_words(p['words'],summary['first_pages'][str(i)]['words']) for i,p in enumerate(pages)]
            row=dict(repetition=repetition,mode=mode,pages=summary['pages'],seconds=summary['wall_seconds'],
                     pps=summary['pages_per_second'],exact=[s['exact'] for s in scores],scores=scores,
                     resources=summary['resources'],max_inflight=summary['max_inflight'])
            assert row['pages']==args.pages and row['max_inflight']<=3
            rows.append(row)
            (out/'comparison.json').write_text(json.dumps(rows,indent=2))
    headings=['Trial','Mode','Pages','Seconds','Pages/s','Exact words by page','Incremental peak GiB','Mean GPU %']
    values=[[r['repetition'],r['mode'],r['pages'],f"{r['seconds']:.2f}",f"{r['pps']:.3f}",r['exact'],
             f"{r['resources']['vram_increment_peak_bytes']/2**30:.2f}",f"{r['resources']['gpu_util_pct_mean']:.1f}"] for r in rows]
    table='<table><tr>'+''.join(f'<th>{h}</th>' for h in headings)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in values)+'</table>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Native dense refinement</title><style>body{font:16px system-ui;margin:40px}td,th{padding:12px;border-bottom:1px solid #ccc}table{border-collapse:collapse;display:block;overflow-x:auto;max-width:100%}</style><h1>Native dense refinement</h1><p>Two sequential trials in reversed order. Same statement pages, FP32 weights, queues and batches. Model loading and one workload warmup excluded; measured time includes decode, inference, ordered JSONL output and drain. Device memory is sampled relative to pre-load idle usage. Exact words use IoU 0.5. Synthetic development fixtures; no local-rotation correction.</p>'+table,encoding='utf-8')
    print(out/'report.html')


if __name__=='__main__':main()
