"""Native accuracy ablation for page direction, deskew and dense refinement."""
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pybaseline.metrics import score_words
from pybaseline.quality_report import overlay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();out=(args.output or ROOT/'pybaseline/results'/('native_orientation_'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve();out.mkdir(parents=True,exist_ok=False)
    fixtures=ROOT/'output/pdf/quality';manifest=json.loads((fixtures/'manifest.json').read_text())
    pages=manifest['pages'];workload=out/'workload.json'
    workload.write_text(json.dumps({'pages':[{'id':p['id'],'image':str((fixtures/p['image']).resolve())} for p in pages]}))
    env=os.environ.copy();runtime=ROOT/'.venv-baseline/Lib/site-packages/onnxruntime/capi'
    env['ORT_DYLIB_PATH']=str(runtime/'onnxruntime.dll');env['PATH']=str(runtime)+os.pathsep+str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib')+os.pathsep+env['PATH']
    provenance={}
    for path in [ROOT/'target/release/throughput.exe',ROOT/'models/page_orientation.onnx',ROOT/'models/page_orientation.json',ROOT/'models/db_resnet34.onnx',ROOT/'models/parseq.onnx',fixtures/'manifest.json']:
        with path.open('rb') as file:provenance[str(path.relative_to(ROOT))]=hashlib.file_digest(file,'sha256').hexdigest()
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    modes={'base':[],'quarter':['--page-orientation'],'deskew':['--page-orientation','--deskew'],'refined':['--page-orientation','--deskew','--dense-refine']}
    rows=[]
    for mode,flags in modes.items():
        folder=out/mode
        subprocess.run([str(ROOT/'target/release/throughput.exe'),'--workload',str(workload),'--output',str(folder),
                        '--pages',str(len(pages)),'--size','1536','--reco-batch','256','--workers','2','--inflight','3',
                        '--arena-mib','6144','--det-arena-mib','4096',*flags],cwd=ROOT,env=env,check=True,timeout=1200)
        summary=json.loads((folder/'summary.json').read_text())
        for i,page in enumerate(pages):
            record=summary['first_pages'][str(i)]
            words=[dict(w,polygon=w.get('quadrilateral',w['polygon'])) for w in record['words']]
            row=dict(mode=mode,page=page['id'],truth_angle=page['page_rotation_deg'],score=score_words(page['words'],words),geometry=record.get('geometry'))
            rows.append(row)
        (out/'accuracy.json').write_text(json.dumps(rows,indent=2))
    table=[]
    for page in pages:
        r=[next(r for r in rows if r['page']==page['id'] and r['mode']==mode) for mode in modes]
        g=r[2]['geometry']
        table.append([page['id'],*[v['score']['exact'] for v in r],g['applied_quarter_deg'],g['skew']['estimated_deg'],g['skew']['applied_deg'],g['skew']['reason']])
    headers=['Page','Base exact','Quarter exact','Deskew exact','Deskew + refinement','Quarter CW','Estimated skew','Applied skew','Decision']
    html_table='<table><tr>'+''.join(f'<th>{s}</th>' for s in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in table)+'</table>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Native page orientation</title><style>body{font:16px system-ui;margin:30px}table{display:block;overflow:auto;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ddd}th{background:#18344e;color:white}</style><h1>Native page orientation and fractional deskew</h1><p>Accuracy ablation, same DB ResNet34 + PARSeq FP32 weights at 1536. Each mode uses one full-page detector pass; refinement adds bounded tile passes. Exact words require one-to-one IoU 0.5 and exact text. Scoring uses original-image quadrilaterals, not their axis-aligned envelopes. Quarter turns use lossless pixel permutations; only fractional skew is resampled. Synthetic development fixtures, not held-out validation. Short-run timing is not a throughput comparison.</p>'+html_table,encoding='utf-8')
    page=next(p for p in pages if p['id']=='statement_0_cw90.5')
    record=next(p for p in summary['first_pages'].values() if p['id']==page['id'])
    words=[dict(w,polygon=w['quadrilateral']) for w in record['words']]
    diagram=overlay(page,words).replace('../../../output/pdf/quality/',os.path.relpath(fixtures,out).replace(os.sep,'/')+'/')
    (out/'overlay.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Original-coordinate overlay</title><style>body{font:16px system-ui}svg{width:1100px;max-width:100%}</style><h2>Original 90.5 degree input: green truth / red native quadrilaterals</h2>'+diagram,encoding='utf-8')
    with (out/'report.html').open('a',encoding='utf-8') as report:
        report.write('<p><a href="overlay.html">Inspect original-coordinate overlay</a></p>')
    print(out/'report.html')


if __name__=='__main__':main()
