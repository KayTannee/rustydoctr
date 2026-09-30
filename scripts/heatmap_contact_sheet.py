"""Side-by-side labelled boundary examples for box merging and heatmap fusion."""
import html,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_public import points
from scripts.report_tiling import diagnose
from scripts.prepare_public_datasets import write

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def records(p):return {r['id']:r for r in map(json.loads,p.open(encoding='utf-8'))}
def main():
    root=ROOT/'pybaseline/results/tiling_dev_v1';body=['<h1>Box merging versus heatmap fusion</h1><p>Same page, crop grid, detector and recognizer weights. Left: guarded box merging. Right: weighted heatmap fusion. Blue is the target label, red the OCR boxes; hover for predicted text. Examples show both gains and losses in strict polygon matching. Selection: first four gains and losses per corpus at crop edges in frozen page order. No hand-picked best cases.</p>'];evidence=[]
    for ds in ['funsd','cord-v2','hiertext']:
        pages=read(root/ds/'manifest.json');a=root/ds/'tiles2048_o128_guarded';b=root/ds/'tiles2048_o128_heatmap/weighted'
        if not all((p/'edge_diagnostics.json').exists() for p in [a,b]):continue
        before=read(a/'edge_diagnostics.json')['pages'];after=read(b/'edge_diagnostics.json')['pages'];ar=records(a/'pages.jsonl');br=records(b/'pages.jsonl');used={'gain':0,'loss':0};body.append('<h2>'+ds+'</h2>')
        for page in pages:
            for x,y in zip(before[page['id']],after[page['id']]):
                if x['exact']==y['exact'] or not (x['crosses_crop_edge'] or y['crosses_crop_edge']):continue
                kind='gain' if y['exact'] else 'loss'
                if used[kind]>=4:continue
                used[kind]+=1;evidence.append(dict(dataset=ds,page=page['id'],kind=kind,before=x,after=y))
                W,H=page['width'],page['height'];q=np.array(x['polygon'])*[W,H];x0,y0=q.min(axis=0)-[60,25];x1,y1=q.max(axis=0)+[60,25];src=html.escape(os.path.relpath(page['image'],root).replace(os.sep,'/'),quote=True);truth=' '.join(f'{a},{b}' for a,b in q)
                body.append('<h3>'+html.escape(x['text']+' — '+kind+' / '+page['id'])+'</h3><div class="pair">')
                for label,record in [('Guarded boxes',ar[page['id']]),('Weighted heatmap',br[page['id']])]:
                    boxes=''
                    for word in record['words']:
                        p=points(word)*[W,H];lo=p.min(axis=0);hi=p.max(axis=0)
                        if hi[0]<x0 or lo[0]>x1 or hi[1]<y0 or lo[1]>y1:continue
                        pts=' '.join(f'{a},{b}' for a,b in p);boxes+=f'<polygon points="{pts}" fill="none" stroke="#ce3427" stroke-width="1.1"><title>{html.escape(word["text"])}</title></polygon>'
                    body.append(f'<div><b>{label}</b><svg viewBox="{x0} {y0} {x1-x0} {y1-y0}"><image href="{src}" width="{W}" height="{H}"/>{boxes}<polygon points="{truth}" fill="none" stroke="#1377cc" stroke-width="1.6"/></svg></div>')
                body.append('</div>')
    body.append('<p><a href="report.html">Full comparison</a> · <a href="fusion_examples.json">Example evidence</a></p>');write(root/'fusion_examples.json',evidence)
    (root/'fusion_examples.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Fusion boundary examples</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px;color:#203247}p{line-height:1.5}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.pair>div{min-width:0;border:1px solid #ccd;padding:10px}svg{width:100%;height:180px}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>'+''.join(body),encoding='utf-8');print('Examples',len(evidence),flush=True)
if __name__=='__main__':main()
