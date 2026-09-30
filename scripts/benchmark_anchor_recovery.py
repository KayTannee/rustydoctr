"""Compare a preserved thin-v1 executable with current native recovery on 23 pages."""
import argparse, hashlib, html, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tune_dense_detection import read,write,environment
from scripts.audit_missing_words import analyse
from pybaseline.metrics import polygon

def report(out):
    runs=read(out/'runs.json'); pages=read(out/'manifest.json')
    old=read(out/'quality_old/summary.json')['first_pages'];new=read(out/'quality_new/summary.json')['first_pages']
    rows=[]
    for i,page in enumerate(pages):
        before=old[str(i)]['words'];after=new[str(i)]['words']
        key=lambda w:(w['text'],tuple(round(v,6) for p in w['polygon'] for v in p))
        oldkeys={key(w) for w in before};newkeys={key(w) for w in after}
        added=[w for w in after if key(w) not in oldkeys];removed=[w for w in before if key(w) not in newkeys];details=[]
        for word in added:
            shape=polygon(word.get('quadrilateral',word['polygon']))
            hits=[t for t in page['words'] if polygon(t['polygon']).area and shape.intersection(polygon(t['polygon'])).area/polygon(t['polygon']).area>=.5]
            details.append(dict(word=word,truth=[t['text'] for t in hits],correct=len(hits)==1 and hits[0]['text']==word['text']))
        count=lambda ws:sum(r['group']=='ordinary' and r['status']=='no_box_overlap' for r in analyse(page,ws))
        rows.append(dict(page=page['id'],added=len(added),correct=sum(d['correct'] for d in details),other=sum(not d['correct'] for d in details),removed=removed,missing_before=count(before),missing_after=count(after),details=details))
    write(out/'quality.json',rows)
    totals={k:sum(r[k] for r in rows) for k in ['added','correct','other','missing_before','missing_after']};totals['removed']=sum(len(r['removed']) for r in rows)
    timed=[r for r in runs if not r['folder'].startswith('quality')];rates={}
    for mode in ['old','new']:
        selected=[r for r in timed if r['mode']==mode]
        if selected:rates[mode]=sum(r['pages'] for r in selected)/sum(r['seconds'] for r in selected)
    if len(rates)==2:rates['change_pct']=(rates['new']/rates['old']-1)*100
    write(out/'comparison.json',dict(quality=totals,throughput=rates))
    def table(headers,rows):return '<div style="overflow:auto"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</table></div>'
    body='<h1>Native one-hop recovery and tile reconciliation</h1><p>23 synthetic page variants. Old and new both enable thin recovery. Same DB ResNet34 + PARSeq models, 1536 full-page detector, 1024 dense tiles, recognition batch 256, two CPU workers, three admitted pages. Model loading and one full-workload warmup excluded; rendering inputs are fixed. Incremental VRAM includes load/warmup relative to pre-load idle. Synthetic evidence is not a real-document accuracy guarantee.</p>'
    body+='<p>'+html.escape(str(totals))+'</p><p>Correct addition requires exact text and at least 50% coverage of exactly one labelled word, not strict IoU. The known partial I remains counted as other.</p>'
    body+=table(['Page','Added','Correct','Other','Removed','Missing before','After'],[[r['page'],r['added'],r['correct'],r['other'],len(r['removed']),r['missing_before'],r['missing_after']] for r in rows])
    body+=table(['Run','Pages','Seconds','Pages/s','Device GiB','Increment GiB','GPU %'],[[r['folder'],r['pages'],round(r['seconds'],2),round(r['pps'],3),round(r['resources']['device_vram_bytes_max']/2**30,2),round(r['resources']['vram_increment_peak_bytes']/2**30,2),round(r['resources']['gpu_util_pct_mean'],1)] for r in runs])
    body+='<h2>Added words</h2><p>Green: verified addition; orange: insufficient box coverage. Blue: intersecting labels. Crops show original page coordinates.</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">'
    for page,row in zip(pages,rows):
        W,H=page['width'],page['height'];src=html.escape(os.path.relpath(page['image'],out).replace(os.sep,'/'),quote=True)
        for detail in row['details']:
            word=detail['word'];points=word.get('quadrilateral',word['polygon']);shape=polygon(points);x0,y0,x1,y1=shape.bounds
            x=max(0,x0*W-90);y=max(0,y0*H-30);cw=min(W-x,(x1-x0)*W+180);ch=min(H-y,(y1-y0)*H+60)
            pts=lambda ps:' '.join(f'{a*W},{b*H}' for a,b in ps)
            label_shapes=''.join(f'<polygon points="{pts(t["polygon"])}" fill="none" stroke="#377bc3" stroke-width="1.5"/>' for t in page['words'] if shape.intersects(polygon(t['polygon'])))
            color='#20854b' if detail['correct'] else '#d17813'
            predicted=list(shape.exterior.coords)
            body+=f'<div style="border:1px solid #ccd;padding:10px;min-width:0"><b>{html.escape(word["text"])}</b> / {html.escape(page["id"])}<svg style="width:100%;height:130px" viewBox="{x} {y} {cw} {ch}"><image href="{src}" width="{W}" height="{H}"/>{label_shapes}<polygon points="{pts(predicted)}" fill="none" stroke="{color}" stroke-width="2"/></svg></div>'
    body+='</div>'
    body+='<p>Pooled measured throughput: '+html.escape(str(rates))+'. Two orders cannot establish statistical significance for small differences.</p><p><a href="quality.json">Word evidence</a> · <a href="runs.json">Timings and binary hashes</a> · <a href="repeat_checks.json">Repeated output checks</a></p>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Native anchor recovery</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#203247}td,th{padding:9px;border:1px solid #ddd}table{border-collapse:collapse}p{line-height:1.5}</style>'+body,encoding='utf-8')
    print(dict(quality=totals,throughput=rates),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--old-binary',type=Path,default=ROOT/'.cache/native-thin-v1/throughput.exe');p.add_argument('--pages',type=int,default=230);p.add_argument('--quality-only',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args();out=a.output.resolve()
    if a.report_only:report(out);return
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'manifest.json').exists():
        pages=read(ROOT/'pybaseline/results/native_thin_v1/manifest.json')+read(ROOT/'pybaseline/results/anchor_holdout_v1/source/manifest.json')
        write(out/'manifest.json',pages);write(out/'workload.json',dict(pages=[dict(id=p['id'],image=p['image']) for p in pages]));write(out/'runs.json',[])
        files=[*[Path(p['image']) for p in pages],*[ROOT/'models'/n for n in ['db_resnet34.onnx','parseq.onnx','metadata.json','page_orientation.onnx','page_orientation.json']]]
        write(out/'provenance.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    pages=read(out/'manifest.json');assert a.pages%len(pages)==0;runs=read(out/'runs.json')
    plans=[('quality_old','old',len(pages)),('quality_new','new',len(pages))]
    if not a.quality_only:plans += [('1_old','old',a.pages),('1_new','new',a.pages),('2_new','new',a.pages),('2_old','old',a.pages)]
    for name,mode,count in plans:
        if any(r['folder']==name for r in runs):continue
        exe=a.old_binary.resolve() if mode=='old' else ROOT/'target/release/throughput.exe'
        cmd=[str(exe),'--workload',str(out/'workload.json'),'--output',str(out/name),'--pages',str(count),'--size','1536','--reco-batch','256','--det-batch','1','--workers','2','--inflight','3','--arena-mib','6144','--det-arena-mib','4096','--page-orientation','--deskew','--dense-refine','--thin-recovery']
        print('Run',name,flush=True);subprocess.run(cmd,cwd=ROOT,env=environment(),check=True,timeout=1800)
        s=read(out/name/'summary.json');assert s['pages']==count and s['max_inflight']<=3
        runs.append(dict(folder=name,mode=mode,pages=count,seconds=s['wall_seconds'],pps=s['pages_per_second'],resources=s['resources'],max_inflight=s['max_inflight'],stage_seconds=s['stage_seconds'],binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()));write(out/'runs.json',runs)
        if (out/'quality_new/summary.json').exists():report(out)
if __name__=='__main__':main()
