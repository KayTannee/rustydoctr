"""Report tile accuracy, edge exposure, duplicates and shifted-grid stability."""
import html,json,os,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon,box
from shapely.strtree import STRtree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_tiling import MODES as BASE_MODES,windows
MODES=dict(BASE_MODES)
MODES.update({k+"_guarded":v for k,v in BASE_MODES.items() if v[2] and v[2][1]})
MODES.update({k+"_heatmap_"+policy:v for k,v in BASE_MODES.items() if v[2] and v[2][1] for policy in ["uniform","weighted"]})
from scripts.score_public import points
from scripts.prepare_public_datasets import write

def diagnose(page,record):
    W,H=page['width'],page['height'];truth=[w for w in page['words'] if not w.get('ignore')]
    ignored=[Polygon(points(w)).buffer(0) for w in page['words'] if w.get('ignore')]
    words=record['words'];ps=[Polygon(points(w)).buffer(0) for w in words]
    ids=[j for j,p in enumerate(ps) if not any(p.area and p.intersection(g).area/p.area>=.5 for g in ignored)]
    words=[words[j] for j in ids];ps=[ps[j] for j in ids];gs=[Polygon(points(w)).buffer(0) for w in truth]
    ious=np.zeros((len(gs),len(ps)));touch=np.zeros(len(gs),dtype=bool);tree=STRtree(ps)
    for i,g in enumerate(gs):
        for j in tree.query(g):
            area=g.intersection(ps[j]).area;touch[i]|=area>0;union=g.area+ps[j].area-area
            if union:ious[i,j]=area/union
    pairs={}
    if gs and ps:
        ii,jj=linear_sum_assignment(-ious);pairs={int(i):int(j) for i,j in zip(ii,jj) if ious[i,j]>=.5}
    tiles=record.get('tiles',[])
    if not tiles:tiles=[(t['x'],t['y'],t['x']+t['width'],t['y']+t['height']) for t in record.get('refinement_tiles',[])]
    rows=[]
    for i,(g,w) in enumerate(zip(gs,truth)):
        x0,y0,x1,y1=np.array(g.bounds)*[W,H,W,H]
        cross=any(((0<a<W and x0<a<x1) or (0<c<W and x0<c<x1)) and max(y0,b)<min(y1,d) or ((0<b<H and y0<b<y1) or (0<d<H and y0<d<y1)) and max(x0,a)<min(x1,c) for a,b,c,d in tiles)
        intact=any(a<=x0 and b<=y0 and c>=x1 and d>=y1 for a,b,c,d in tiles)
        seam=any(x0<t['owner_end']<x1 and y0<t['y']+t['height'] and y1>t['y'] for t in record.get('refinement_tiles',[]) if 0<t['owner_end']<W)
        rows.append(dict(index=i,text=w['text'],polygon=w['polygon'],crosses_crop_edge=cross,intact_tile=intact,ownership_seam=seam,detected=i in pairs,exact=i in pairs and words[pairs[i]]['text']==w['text'],no_overlap=not bool(touch[i]),duplicates=max(0,int((ious[i]>=.5).sum())-1)))
    def count(rows):return dict(truth=len(rows),detected=sum(r['detected'] for r in rows),exact=sum(r['exact'] for r in rows),no_overlap=sum(r['no_overlap'] for r in rows),duplicates=sum(r['duplicates'] for r in rows))
    groups={'all':count(rows),'crop_edge':count([r for r in rows if r['crosses_crop_edge']]),'cut_in_every_tile':count([r for r in rows if r['crosses_crop_edge'] and not r['intact_tile']]),'ownership_seam':count([r for r in rows if r['ownership_seam']])}
    return rows,groups

def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def main():
    out=ROOT/'pybaseline/results/tiling_dev_v1';body=['<h1>Resolution and overlapping tiles</h1><p>350 frozen development images: 50 FUNSD training forms, all 100 CORD validation receipts, 200 HierText validation images selected by SHA-256 of image ID. No test images reused. Same native DB ResNet34/PARSeq weights throughout; orientation, deskew, line alternatives and thin recovery disabled.</p><details><summary>Protocol and limitations</summary><p>Full tiles have 1024×1024 detector inputs. 1536 or 2048 denotes the effective page long side: original crops span 2/3 or 1/2 of that long side, respectively, then native preprocessing resizes to 1024. This may upscale low-resolution scans; it adds no source detail. Overlap is 0, 128 or 256 detector pixels; shift moves both grid axes by half a stride. Edge crops retain square size with white padding.</p><p>Selective1024 uses the existing native small-text band selector and seam reconciliation, not recognition-confidence gating. Full tiling is an experimental Python adapter: it recognizes each crop, maps to page coordinates and merges with text-independent IoU ≥0.5 NMS, preferring observations farther from interior crop edges. It is not the native selective merge algorithm. Raw tile detections are retained. Guarded variants reuse those detections: suppress an observation within eight detector pixels of an interior edge only when another tile offers at least sixteen pixels of context around it, then apply the same NMS. This may remove fragments but can also lose words if the alternative tile failed to detect them; both effects count in the scores. Heatmap variants instead blend sigmoid probabilities before DB thresholding, morphology and box extraction. Uniform fusion averages all observations; weighted fusion tapers linearly over 128 pixels at interior tile edges and normalizes accumulated weights. Rectangular page maps avoid square padding. Recognition runs on retained boxes from the original image, once per fusion policy, with normal wide-word splitting. Both policies share detector inference, so have no separate throughput result. Elapsed timings include cold model loading, Python merging and redundant recognition, so they are not optimized production throughput.</p></details>'];totals=[];cases=[]
    for ds in ['funsd','cord-v2','hiertext']:
        folder=out/ds;pages=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));byid={p['id']:p for p in pages};rows=[];edge_rows=[];details={}
        for mode in MODES:
            run=folder/mode
            if "_heatmap_" in mode:
                base,policy=mode.rsplit("_heatmap_",1);run=folder/(base+"_heatmap")/policy
            if not (run/'scores.json').exists():continue
            score=json.loads((run/'scores.json').read_text(encoding='utf-8'));summary=json.loads((run/'summary.json').read_text(encoding='utf-8'));m=score['metrics']
            keys=['Det-Fscore','E2E-Fscore'] if ds=='hiertext' else ['detection_f1','end_to_end_f1']
            rows.append([mode,*[f'{100*m[k]:.2f}' for k in keys],summary['crop_passes'],f"{summary['wall_seconds']:.1f}" if summary['wall_seconds'] is not None else "reused"])
            if (run/'edge_diagnostics.json').exists():
                cached=json.loads((run/'edge_diagnostics.json').read_text(encoding='utf-8'));groups=cached['groups'];detail=cached['pages']
            else:
                groups={};detail={}
                for line in (run/'pages.jsonl').open(encoding='utf-8'):
                    record=json.loads(line);d,g=diagnose(byid[record['id']],record);detail[record['id']]=d
                    for name,v in g.items():groups.setdefault(name,Counter()).update(v)
                write(run/'edge_diagnostics.json',dict(groups={k:dict(v) for k,v in groups.items()},pages=detail))
            details[mode]=detail
            for group in ['all','crop_edge','cut_in_every_tile','ownership_seam']:
                g=groups[group]
                if g['truth']:edge_rows.append([mode,group,*[g[k] for k in ['truth','detected','exact','no_overlap','duplicates']]])
            totals.append(dict(dataset=ds,mode=mode,scores=score,summary=summary,edges={k:dict(v) for k,v in groups.items()}))
        body.append('<h2>'+ds+'</h2>'+table(['Mode','Detection F1 %','Exact OCR F1 %','Crop/page passes','Cold elapsed s'],rows))
        body.append('<details><summary>Edge and omission diagnostics</summary><p>Polygon assignment matching at IoU ≥0.5, ignoring illegible regions; distinct from HierText official aggregate. Crop-edge groups change with grid placement, so compare whole-page metrics and paired word changes too. Duplicate count is extra predictions matching the same label. “Cut in every tile” means the label crosses a crop boundary and no tile fully contains its bounding envelope.</p>'+table(['Mode','Group','Labels','Detected','Exact','No overlap','Extra duplicates'],edge_rows)+'</details>')
        for before,after in [('tiles2048_o0','tiles2048_o128'),('tiles2048_o128','tiles2048_o256'),('tiles2048_o128','tiles2048_o128_shift'),('plain1024','selective1024'),('tiles2048_o0','tiles2048_o128_guarded'),('tiles2048_o128_guarded','tiles2048_o256_guarded'),('tiles2048_o128_guarded','tiles2048_o128_shift_guarded'),('tiles2048_o128_guarded','tiles2048_o128_heatmap_weighted'),('tiles2048_o128_heatmap_weighted','tiles2048_o256_heatmap_weighted'),('tiles2048_o128_heatmap_weighted','tiles2048_o128_shift_heatmap_weighted')]:
            if before not in details or after not in details:continue
            counts=Counter()
            for page in pages:
                for a,b in zip(details[before][page['id']],details[after][page['id']]):
                    assert a['index']==b['index'] and a['text']==b['text']
                    counts['gained_exact']+=not a['exact'] and b['exact'];counts['lost_exact']+=a['exact'] and not b['exact']
                    counts['gained_detection']+=not a['detected'] and b['detected'];counts['lost_detection']+=a['detected'] and not b['detected']
                    if (a['crosses_crop_edge'] or b['crosses_crop_edge'] or b['ownership_seam']) and a['detected']!=b['detected']:
                        cases.append(dict(dataset=ds,page=page['id'],before=before,after=after,word=a,after_word=b))
            body.append('<p>'+html.escape(before+' → '+after+': '+str(dict(counts)))+'</p>')
    write(out/'results.json',totals);write(out/'edge_changes.json',cases)
    summary=[]
    for mode in ['plain1024','plain1536','selective1024','tiles1536_o128_guarded','tiles1536_o128_heatmap_weighted','tiles2048_o128_guarded','tiles2048_o128_heatmap_weighted','tiles2048_o256_heatmap_weighted','tiles2048_o128_shift_heatmap_weighted']:
        row=[mode]
        for ds in ['funsd','cord-v2','hiertext']:
            match=next((r for r in totals if r['dataset']==ds and r['mode']==mode),None)
            row.append(f"{100*match['scores']['metrics']['E2E-Fscore' if ds=='hiertext' else 'end_to_end_f1']:.2f}" if match else 'pending')
        summary.append(row)
    body.insert(1,'<h2>Findings</h2><p>Higher resolution recovers substantially more tiny text, but that does not guarantee better whole-page F1. On HierText, labels below eight pixels high at 1024 improve from 14.3% exact recall with plain 1024 to 43.2% with 2048/128 weighted fusion; zero-overlap omissions fall from 2,852 to 1,000. Larger text and false positives offset some gains. CORD has only five labels below twelve pixels in this cohort, and all FUNSD originals have a 1,000-pixel long side.</p><p>At effective 1536, weighted heatmap fusion improves on guarded box merging in all three corpora. It does not establish a universal replacement for the whole-page pass. Next: isolate the possible gain on labelled small-text regions, then develop a selector that applies refinement locally and measures damage to already-correct words. Keep the public corpora as regression checks.</p>')
    body.insert(2,'<h2>Exact word F1 (%)</h2>'+table(['Configuration','FUNSD','CORD v2','HierText'],summary)+'<p>Development results, not published test scores. Full tables below include uniform blending, unguarded merging, detector scores and paired losses. <a href="fusion_examples.html">Side-by-side fusion examples</a> · <a href="size_strata.html">Accuracy by labelled word size</a>.</p>')
    body.append('<h2>Boundary examples</h2><p>First four detection losses and gains per comparison and dataset, in frozen page order. Blue: label; orange: crop edges from the destination grid. These are diagnostic examples, not proof that an edge caused every change.</p><div class="cards">')
    used=Counter()
    for case in cases:
        kind='gain' if case['after_word']['detected'] else 'loss';key=(case['dataset'],case['before'],case['after'],kind)
        if used[key]>=4:continue
        used[key]+=1
        page=next(p for p in json.loads((out/case['dataset']/'manifest.json').read_text(encoding='utf-8')) if p['id']==case['page']);W,H=page['width'],page['height'];q=np.array(case['word']['polygon'])*[W,H];x,y=q.min(axis=0)-[60,30];xx,yy=q.max(axis=0)+[60,30];src=os.path.relpath(page['image'],out).replace(os.sep,'/');pts=' '.join(f'{a},{b}' for a,b in q)
        tile=MODES[case['after']][2];rects=''
        if tile:
            for a,b,c,d in windows(W,H,*tile):rects+=f'<rect x="{a}" y="{b}" width="{c-a}" height="{d-b}" fill="none" stroke="#d76a19" stroke-width="2"/>'
        body.append(f'<div class="card"><b>{html.escape(case["word"]["text"])}: {kind}</b><p>{html.escape(case["dataset"]+" / "+case["before"]+" → "+case["after"])}</p><svg viewBox="{x} {y} {xx-x} {yy-y}"><image href="{html.escape(src,quote=True)}" width="{W}" height="{H}"/>{rects}<polygon points="{pts}" fill="none" stroke="#1672c1" stroke-width="2"/></svg></div>')
    body.append('</div><p><a href="results.json">All results</a> · <a href="edge_changes.json">Paired edge changes</a></p>')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Tile accuracy</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:20px;color:#203247}p{line-height:1.5}.scroll{overflow:auto}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccd}th{background:#e1edf5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.card{min-width:0;overflow-wrap:anywhere;border:1px solid #ccd;padding:10px}svg{width:100%;height:140px}summary{cursor:pointer;margin:15px 0}</style>'+''.join(body),encoding='utf-8');print(out/'report.html',flush=True)
if __name__=='__main__':main()
