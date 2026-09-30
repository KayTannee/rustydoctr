"""Accuracy/error audit for bounded line-guided crops; no sustained timing claims."""
import argparse
from datetime import datetime
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pybaseline.metrics import score_words,polygon
from pybaseline.quality_report import table,overlay


def heldout(folder):
    """New unruled layouts, both local quarter-turn directions, separate from tuning data."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from scripts.generate_document_fixtures import transform
    folder.mkdir(parents=True,exist_ok=True)
    pages=[]
    for layout in range(2):
        w,h=1800,2400;image=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(image);words=[]
        font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/('arial.ttf' if layout==0 else 'times.ttf')
        def line(text,x,y,size,region,angle=0):
            font=ImageFont.truetype(str(font_path),size);ascent,descent=font.getmetrics()
            width=int(draw.textlength(text,font=font))+20
            strip=Image.new('RGB',(width,ascent+descent+16),'white');pen=ImageDraw.Draw(strip);cursor=8.;local=[]
            for word in text.split():
                extent=pen.textlength(word,font=font)
                pen.text((cursor,8),word,font=font,fill='black',anchor='la')
                # Use actual glyph extents for this independent holdout, unlike font-metric development labels.
                box=pen.textbbox((cursor,8),word,font=font,anchor='la')
                x0,y0,x1,y1=box
                local.append(dict(text=word,polygon=[[x0/width,y0/strip.height],[x1/width,y0/strip.height],[x1/width,y1/strip.height],[x0/width,y1/strip.height]],region=region))
                cursor+=extent+pen.textlength(' ',font=font)
            raster,mapped,_,_=transform(np.array(strip),local,[],angle)
            sh,sw=raster.shape[:2];assert x+sw<=w and y+sh<=h
            image.paste(Image.fromarray(raster),(x,y))
            for word in mapped:
                word['polygon']=[[(px*sw+x)/w,(py*sh+y)/h] for px,py in word['polygon']]
                words.append(word)
        line('MONTHLY SERVICE RECORD',100,100,56,'title')
        line('Client: A Ipsum   Reference: I - 1704',100,210,32,'labels')
        for row in range(16):
            line(['It was cold outside I put a coat on.','I retain a copy - a record is provided.','A payment - I agree; a balance remains.','Lorem ipsum dolor sit amet.'][row%4],100,340+row*67,30 if layout==0 else 27,'body')
        line('I a A i l 1 O 0 - _ I a',100,1500,34,'characters')
        line('REFERENCE I A 7391',1570,350,28,'local90',90)
        line('ACCOUNT a I 4826',1640,1100,28,'local270',270)
        line('Amount I a',1150,1600,30,'local45',45)
        for row in range(5):line('I retain a copy; a statement is supplied - I agree.',100,1840+row*36,22,'small')
        for angle in [0,.7,90,180]:
            raster,truth,_,_=transform(np.array(image),words,[],angle)
            name=f'holdout_{layout}_cw{angle:g}';path=folder/f'{name}.png';Image.fromarray(raster).save(path)
            pages.append(dict(id=name,image=str(path.resolve()),width=raster.shape[1],height=raster.shape[0],words=truth,page_rotation_deg=angle,dataset='holdout_ink_labels'))
    return pages


def errors(page,words):
    pairs=dict(score_words(page['words'],words,.25,True)['pairs'])
    shapes=[polygon(w['polygon']) for w in words];truth=[polygon(w['polygon']) for w in page['words']]
    from shapely.strtree import STRtree
    tree=STRtree(shapes);gt_tree=STRtree(truth)
    result=[]
    for i,g in enumerate(truth):
        t=page['words'][i];j=pairs.get(i)
        if j is not None:category='exact' if t['text']==words[j]['text'] else 'matched_wrong_text'
        else:
            overlapping=[int(k) for k in tree.query(g) if g.intersection(shapes[k]).area>0]
            covering=[k for k in overlapping if g.area and g.intersection(shapes[k]).area/g.area>=.5]
            merged=any(sum(truth[q].area>0 and shapes[k].intersection(truth[q]).area/truth[q].area>=.5 for q in gt_tree.query(shapes[k]))>1 for k in covering)
            category='merged_coverage' if merged else 'overlap_without_match' if overlapping else 'no_overlapping_detection'
        result.append(dict(truth=t['text'],predicted=words[j]['text'] if j is not None else None,region=t['region'],category=category))
    return result


def report(out,pages):
    rows=[];details=[];changed=[];transitions=[]
    summaries={mode:json.loads((out/mode/'summary.json').read_text(encoding='utf-8')) for mode in ['base','guided']}
    for i,page in enumerate(pages):
        correct={}
        for mode in summaries:
            record=summaries[mode]['first_pages'][str(i)]
            words=[dict(w,polygon=w.get('quadrilateral',w['polygon'])) for w in record['words']]
            audit=errors(page,words);score=score_words(page['words'],words)
            correct[mode]={j for j,k in score_words(page['words'],words,.5,True)['pairs'] if page['words'][j]['text']==words[k]['text']}
            trials=[w for w in words if w.get('crop_decision')]
            accepted=[w for w in trials if w['crop_decision']['selected_rotation_deg'] is not None]
            rows.append(dict(page=page['id'],dataset=page['dataset'],mode=mode,score=score,
                             categories=dict(Counter(a['category'] for a in audit)),
                             singles=dict(Counter(a['category'] for a in audit if a['truth']=='I')),
                             trial_words=len(trials),extra_crops=sum(len(w['crop_decision']['candidates']) for w in trials),accepted=len(accepted)))
            details.append(dict(page=page['id'],mode=mode,errors=audit))
            for w in accepted:changed.append(dict(page=page['id'],**w['crop_decision'],selected_text=w['text']))
        transitions.append(dict(page=page['id'],gained=len(correct['guided']-correct['base']),lost=len(correct['base']-correct['guided'])))
    (out/'accuracy.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'errors.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    (out/'changes.json').write_text(json.dumps(changed,indent=2),encoding='utf-8')
    (out/'transitions.json').write_text(json.dumps(transitions,indent=2),encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Line-guided crop audit</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#24384c}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}th{background:#e4edf5}svg{width:100%;max-width:800px}summary{cursor:pointer;padding:12px}</style><h1>Line-guided crop audit</h1><p>Same DB ResNet34 + PARSeq FP32, 1536 detector; page orientation, deskew and dense refinement enabled in both runs. Guided mode adds at most 32 recognition crops and 1 million source crop pixels per page. No extra detection pass. Accuracy experiment; these short runs do not establish throughput.</p><p>Strict exact words: one-to-one IoU 0.5 and exact text. Error categories use IoU 0.25 and are geometric evidence, not definitive root causes. Development labels use font metrics; new holdout labels use glyph extents, so compare modes within each set rather than comparing their absolute rates.</p>']
    totals=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        for mode in summaries:
            selected=[r for r in rows if r['dataset']==dataset and r['mode']==mode]
            totals.append([dataset,mode,sum(r['score']['truth'] for r in selected),sum(r['score']['exact'] for r in selected),sum(r['extra_crops'] for r in selected),sum(r['accepted'] for r in selected)])
    parts.append(table(['Set','Mode','Truth','Exact .5','Extra crops','Accepted alternatives'],totals))
    parts.append('<p>Strict exact-word changes: '+str(sum(t['gained'] for t in transitions))+' gained; '+str(sum(t['lost'] for t in transitions))+' lost. Alternative confidence is not proof of correctness.</p>')
    parts.append('<p>The new pages exposed ambiguous opposite-direction predictions in the initial trial. They informed the final abstention rule and are now regression fixtures, not untouched holdout validation.</p>')
    parts.append('<h2>All ground-truth I: error evidence at IoU 0.25</h2>'+table(['Set','Mode','Category','Count'],[[dataset,mode,k,v] for dataset in sorted({r['dataset'] for r in rows}) for mode in summaries for k,v in sum((Counter(r['singles']) for r in rows if r['dataset']==dataset and r['mode']==mode),Counter()).items()]))
    parts.append('<h2>Every page</h2>'+table(['Page','Mode','Exact .5','Matched .5','Crop trials','Accepted'],[[r['page'],r['mode'],r['score']['exact'],r['score']['matched'],r['extra_crops'],r['accepted']] for r in rows]))
    parts.append('<h2>Locally rotated words at IoU 0.25</h2>'+table(['Region','Mode','Truth','Exact','Wrong text','Without match'],[[region,mode,len(items),sum(e['category']=='exact' for e in items),sum(e['category']=='matched_wrong_text' for e in items),sum(e['predicted'] is None for e in items)] for region in ['local90','local270','local45'] for mode in summaries for items in [[e for d in details if d['mode']==mode for e in d['errors'] if e['region']==region]]]))
    parts.append('<h2>Accepted alternatives (not necessarily correct)</h2>'+table(['Page','Evidence','Before','After','Rotation','Confidence before'],[[r['page'],r['evidence'],r['original_text'],r['selected_text'],r['selected_rotation_deg'],round(r['original_confidence'],4)] for r in changed]))
    for index,page in enumerate(pages):
        record=summaries['guided']['first_pages'][str(index)]
        for word in record['words']:
            if word.get('crop_decision',{}).get('selected_rotation_deg') is None:continue
            prediction=dict(word,polygon=word.get('quadrilateral',word['polygon']))
            x0,y0,x1,y1=polygon(prediction['polygon']).bounds
            x0=max(0,x0*page['width']-100);y0=max(0,y0*page['height']-60)
            width=min(page['width']-x0,(x1*page['width']-x0)+100);height=min(page['height']-y0,(y1*page['height']-y0)+60)
            svg=overlay(page,[prediction]).replace(f'viewBox="0 0 {page["width"]} {page["height"]}"',f'viewBox="{x0} {y0} {width} {height}"')
            svg=svg.replace('../../../output/pdf/quality/'+page['image'],os.path.relpath(page['image'],out).replace(os.sep,'/'))
            parts.append(f'<details><summary>Changed crop on {html.escape(page["id"])}: {html.escape(word["crop_decision"]["original_text"])} to {html.escape(word["text"])}</summary>{svg}</details>')
    for index in [0,len(pages)-4]:
        page=pages[index];record=summaries['guided']['first_pages'][str(index)]
        words=[dict(w,polygon=w.get('quadrilateral',w['polygon'])) for w in record['words']]
        svg=overlay(page,words).replace('../../../output/pdf/quality/'+page['image'],os.path.relpath(page['image'],out).replace(os.sep,'/'))
        parts.append(f'<details><summary>{html.escape(page["id"])}: green truth, red prediction</summary>{svg}</details>')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8')
    print(json.dumps(totals));print(out/'report.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only and args.output is None:parser.error('--report-only requires --output')
    out=(args.output or ROOT/'pybaseline/results'/('line_orientation_'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    if args.report_only:return report(out,json.loads((out/'manifest.json').read_text(encoding='utf-8')))
    out.mkdir(parents=True,exist_ok=False)
    fixtures=ROOT/'output/pdf/quality';pages=json.loads((fixtures/'manifest.json').read_text(encoding='utf-8'))['pages']
    pages=[dict(p,image=str((fixtures/p['image']).resolve()),dataset='development') for p in pages]
    pages+=heldout(out/'fixtures')
    (out/'manifest.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
    workload=out/'workload.json';workload.write_text(json.dumps({'pages':[{'id':p['id'],'image':p['image']} for p in pages]}))
    paths=[ROOT/'target/release/throughput.exe',ROOT/'models/db_resnet34.onnx',ROOT/'models/parseq.onnx',ROOT/'models/page_orientation.onnx',out/'manifest.json',*[Path(p['image']) for p in pages]]
    provenance={}
    for p in paths:
        with p.open('rb') as file:provenance[str(p)]=hashlib.file_digest(file,'sha256').hexdigest()
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    env=os.environ.copy();runtime=ROOT/'.venv-baseline/Lib/site-packages/onnxruntime/capi'
    env['ORT_DYLIB_PATH']=str(runtime/'onnxruntime.dll');env['PATH']=str(runtime)+os.pathsep+str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib')+os.pathsep+env['PATH']
    for mode,flags in [('base',[]),('guided',['--line-guided-orientation'])]:
        subprocess.run([str(ROOT/'target/release/throughput.exe'),'--workload',str(workload),'--output',str(out/mode),'--pages',str(len(pages)),
                        '--size','1536','--reco-batch','256','--workers','2','--inflight','3','--arena-mib','6144','--det-arena-mib','4096',
                        '--page-orientation','--deskew','--dense-refine',*flags],cwd=ROOT,env=env,check=True,timeout=1200)
    report(out,pages)

if __name__=='__main__':main()

