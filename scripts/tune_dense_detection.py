"""Cache native detector maps and tune only dense-tile postprocessing."""
import argparse
from collections import Counter
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
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pybaseline.metrics import score_words
from pybaseline.quality_report import table,overlay

BASE=dict(bin_thresh=.3,box_thresh=.1,unclip_ratio=1.5)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def environment():
    env=os.environ.copy();runtime=ROOT/'.venv-baseline/Lib/site-packages/onnxruntime/capi'
    env['ORT_DYLIB_PATH']=str(runtime/'onnxruntime.dll');env['PATH']=str(runtime)+os.pathsep+str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib')+os.pathsep+env['PATH'];return env
def call(args):subprocess.run([str(ROOT/'target/release/refinement_probe.exe'),*map(str,args)],cwd=ROOT,env=environment(),check=True,timeout=1200)


def new_pages(folder):
    from scripts.generate_document_fixtures import transform
    folder.mkdir();pages=[]
    text=('I agree a copy is supplied. Please retain this record for a future enquiry. '
          'A payment of 104.25 was received on 18/09/2026; reference AB-1402. '
          'If I request a change, a revised statement will follow. We apply no additional fee - I confirm receipt. ')
    for i,(face,size,gap,tracking) in enumerate([('arial.ttf',22,7,0),('times.ttf',24,8,.4),('cour.ttf',23,10,0),('verdana.ttf',22,5,.6)]):
        width,height=2481,3508;image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image);words=[]
        path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/face
        def line(tokens,x,y,size,region,gap=10,tracking=0):
            font=ImageFont.truetype(str(path),size);_,top,_,bottom=font.getbbox('Hg',anchor='ls')
            for word in tokens:
                advance=float(font.getlength(word))+max(0,len(word)-1)*tracking
                if tracking:
                    cursor=x
                    for ch in word:draw.text((cursor,y),ch,font=font,fill='black',anchor='ls');cursor+=float(font.getlength(ch))+tracking
                else:draw.text((x,y),word,font=font,fill='black',anchor='ls')
                words.append(dict(text=word,polygon=[[x/width,(y+top)/height],[(x+advance)/width,(y+top)/height],[(x+advance)/width,(y+bottom)/height],[x/width,(y+bottom)/height]],region=region))
                x+=advance+gap
        line('CUSTOMER SERVICE SUMMARY'.split(),150,180,65,'title',20)
        line('Account AB-1402 Period September 2026'.split(),150,320,38,'labels',18)
        for row in range(12):line(f'{row+1:02d} Payment received Reference X{row+100} Amount 104.25'.split(),150,550+row*105,34,'body',18)
        line('Terms and acknowledgement'.split(),150,2660,35,'legal_heading',12)
        font=ImageFont.truetype(str(path),size);tokens=(text*20).split();cursor=0
        for row in range(19):
            values=[];used=0
            while cursor<len(tokens):
                word=tokens[cursor];advance=float(font.getlength(word))+max(0,len(word)-1)*tracking
                if values and used+advance>2160:break
                values.append(word);used+=advance+gap;cursor+=1
            line(values,150,2720+row*(size+8),size,'legal',gap,tracking)
        for angle in [0,.6]:
            raster,truth,_,_=transform(np.array(image),words,[],angle);name=f'dense_validation_{i}_cw{angle:g}';file=folder/f'{name}.png';Image.fromarray(raster).save(file)
            pages.append(dict(id=name,image=str(file.resolve()),words=truth,width=raster.shape[1],height=raster.shape[0],page_rotation_deg=angle,font=face,font_size_px=size,word_gap_px=gap,tracking_px=tracking))
    return pages


def cache(folder,pages):
    folder.mkdir();write(folder/'manifest.json',pages)
    write(folder/'workload.json',dict(pages=[dict(id=p['id'],image=p['image']) for p in pages]))
    call(['--workload',folder/'workload.json','--cache',folder/'cache'])
def map_words(words,info,page):
    matrix=np.array(info['geometry']['corrected_to_original']);result=[]
    for word in words:
        (x0,y0),(x1,y1)=word['polygon'];quad=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])*[info['width'],info['height']]
        quad=np.c_[quad,np.ones(4)]@matrix.T
        result.append(dict(word,polygon=(quad/[page['width'],page['height']]).tolist()))
    return result
def scores(truth,words):
    result={}
    for threshold in [.5,.25]:
        score=score_words(truth,words,threshold,True);pairs=dict(score.pop('pairs'))
        small=[i for i,w in enumerate(truth) if len(w['text'])<=2]
        score.update(short_truth=len(small),short_matched=sum(i in pairs for i in small),short_exact=sum(i in pairs and truth[i]['text']==words[pairs[i]]['text'] for i in small))
        result[str(threshold)]=score
    return result
def postprocess(folder,grid):
    write(folder/'grid.json',grid);call(['--cache',folder/'cache','--grid',folder/'grid.json','--output',folder/'boxes.json'])
    infos={r['id']:r for r in read(folder/'cache/cache.json')};pages={p['id']:p for p in read(folder/'manifest.json')};rows=read(folder/'boxes.json')
    for row in rows:row['scores']=scores(pages[row['page']]['words'],map_words(row['words'],infos[row['page']],pages[row['page']]))
    write(folder/'detection_scores.json',[{k:v for k,v in r.items() if k!='words'} for r in rows]);return rows
def totals(rows,setting,threshold='.5'):
    key='0.5' if threshold=='.5' else threshold
    return dict(sum((Counter(r['scores'][key]) for r in rows if r['setting']==setting),Counter()))
def choose(rows,grid,baseline,shortlist):
    base={r['page']:r for r in rows if r['setting']==baseline}
    eligible=[i for i in shortlist if all(r['scores']['0.5']['exact']>=base[r['page']]['scores']['0.5']['exact'] for r in rows if r['setting']==i) and totals(rows,i,'0.25')['exact']>=totals(rows,baseline,'0.25')['exact']]
    rank=lambda i:(totals(rows,i)['exact'],totals(rows,i,'0.25')['exact'],i==baseline)
    default_choice=max(eligible,key=rank)
    # Validate the strongest challenger even if no development setting beats baseline.
    winner=default_choice if default_choice!=baseline else max([i for i in shortlist if i!=baseline],key=rank)
    return dict(baseline=baseline,shortlist=shortlist,winner=winner,candidate=grid[winner],development_eligible=winner in eligible,default_choice=default_choice)
def recognize(folder,rows,settings):
    # The detector is not loaded in this phase. Use the existing docTR recognizer decoder.
    import cv2,torch,onnxruntime as ort
    from doctr.models.preprocessor import PreProcessor
    from doctr.models.recognition.parseq.pytorch import PARSeqPostProcessor
    from doctr.models.recognition.predictor._utils import split_crops,remap_preds
    from doctr.utils.geometry import extract_crops
    torch.set_num_threads(1);cv2.setNumThreads(1);ort.set_default_logger_severity(3);ort.preload_dlls(directory=str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib'))
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
    session=ort.InferenceSession(str(ROOT/'models/parseq.onnx'),sess_options=opts,providers=[('CUDAExecutionProvider',{'use_tf32':'0','cudnn_conv_algo_search':'HEURISTIC','gpu_mem_limit':str(3*2**30)})]);assert session.get_providers()[0]=='CUDAExecutionProvider'
    meta=read(ROOT/'models/metadata.json')['parseq'];pre=PreProcessor((32,128),256,preserve_aspect_ratio=True,symmetric_pad=False,mean=meta['mean'],std=meta['std']);post=PARSeqPostProcessor(vocab=meta['vocab'])
    infos={r['id']:r for r in read(folder/'cache/cache.json')};results=[]
    for page in read(folder/'manifest.json'):
        info=infos[page['id']];image=np.array(Image.open(folder/'cache'/info['image']).convert('RGB'));memo={};count=0
        for row in [r for r in rows if r['page']==page['id'] and r['setting'] in settings]:
            bounds=np.array([np.array(w['polygon']).reshape(4) for w in row['words']],dtype=np.float32)
            keys=[tuple(np.round(b*[info['width'],info['height'],info['width'],info['height']]).astype(int)) for b in bounds]
            missing={k:i for i,k in enumerate(keys) if k not in memo};indices=list(missing.values());count+=len(indices)
            if indices:
                crops=extract_crops(image,bounds[indices]);crops,mapping,remap=split_crops(crops,8,6,.5);pred=[]
                for batch in pre(crops):pred.extend(post(torch.from_numpy(session.run(None,{'input':batch.numpy()})[0])))
                if remap:pred=remap_preds(pred,mapping,.5)
                for key,(text,confidence) in zip(missing,pred):memo[key]=(text,float(confidence))
            for word,key in zip(row['words'],keys):word['text'],word['confidence']=memo[key]
            words=map_words(row['words'],info,page)
            results.append(dict(page=page['id'],setting=row['setting'],params=row['params'],words=words,scores=scores(page['words'],words)))
        print('Recognized',page['id'],count,'unique pixel crops',flush=True)
        write(folder/'recognition.json',results)
    return results


def report(out):
    selection=read(out/'selection.json');dev=read(out/'development/recognition.json');val=read(out/'validation/recognition.json');grid=read(out/'development/grid.json')
    baseline_rows={r['page']:r for r in val if r['setting']==0}
    validation_ok=all(r['scores']['0.5']['exact']>=baseline_rows[r['page']]['scores']['0.5']['exact'] for r in val if r['setting']==1) and totals(val,1,'0.25')['exact']>=totals(val,0,'0.25')['exact']
    promote=selection['development_eligible'] and validation_ok
    write(out/'recommendation.json',dict(promote=promote,dense_detection=selection['candidate'] if promote else BASE,development_eligible=selection['development_eligible'],validation_eligible=validation_ok,validation_baseline=totals(val,0),validation_challenger=totals(val,1)))
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Dense tile parameter sweep</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 22px;color:#203247}p{line-height:1.5}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#dfebf5}svg{width:100%;max-width:800px}summary{padding:12px;cursor:pointer}</style><h1>Dense-tile parameter sweep</h1><p>Native detector/preprocessing/postprocessing. Full-page size 1536 and full-page thresholds stay fixed. Only the two 1024 dense tiles vary. Cached FP32 probability maps avoid repeated detector inference. The recognizer is the existing PARSeq ONNX model with docTR decoding; native end-to-end confirmation is recorded separately. No language correction.</p><p>Three existing upright statements choose the shortlist and candidate. Four new dense layouts, each upright and at 0.6 degrees, validate that frozen choice. Eight variants are four independent layouts. New labels use font advance and a common line height. Results are synthetic and do not establish production accuracy.</p>']
    parts.append('<p><b>'+('Candidate passes the measured accuracy gates.' if promote else 'Retain the current defaults: no safe improvement established.')+'</b> Best development alternative: '+str(totals(dev,selection['winner'])['exact'])+' versus '+str(totals(dev,selection['baseline'])['exact'])+' exact words. Frozen validation: '+str(totals(val,1)['exact'])+' versus '+str(totals(val,0)['exact'])+'. These are accuracy runs, not a sustained throughput benchmark.</p>')
    rows=[]
    for index in selection['shortlist']:
        t=totals(dev,index);r=totals(dev,index,'0.25');p=grid[index]
        rows.append([index,p['bin_thresh'],p['unclip_ratio'],t.get('exact',0),t.get('matched',0),t.get('merge_candidates',0),r.get('exact',0),r.get('short_exact',0)])
    parts.append('<h2>Development recognition</h2>'+table(['Setting','Pixel threshold','Expansion','Exact .5','Matched .5','Merge boxes','Exact .25','Short exact .25'],rows))
    detection=read(out/'development/detection_scores.json');grid_rows=[]
    for threshold in [.2,.3,.4,.5,.6]:
        cells=[threshold]
        for expansion in [.75,1.,1.25,1.5,1.75]:
            index=next(i for i,p in enumerate(grid) if p['bin_thresh']==threshold and p['unclip_ratio']==expansion)
            t=totals(detection,index);cells.append(f"{t['matched']} / {t['merge_candidates']}")
        grid_rows.append(cells)
    parts.append('<h2>All 25 CPU settings</h2><p>Cells show matched boxes at IoU 0.5 / merge candidates across the three development pages. Box acceptance stays at 0.1; morphology stays 3x3. A high match count does not guarantee readable crops.</p>'+table(['Pixel threshold / expansion',.75,1.,1.25,1.5,1.75],grid_rows))
    parts.append('<p>Frozen challenger: '+html.escape(json.dumps(selection['candidate']))+'. Development eligibility requires no page to lose strict exact words and no total relaxed-score loss. Challenger eligible: '+str(selection['development_eligible'])+'. A challenger is still measured on validation when it fails this gate; this does not promote it to a default.</p>')
    values=[]
    for page in read(out/'validation/manifest.json'):
        rs=[next(r for r in val if r['page']==page['id'] and r['setting']==s) for s in [0,1]]
        values.append([page['id'],*[r['scores']['0.5']['exact'] for r in rs],*[r['scores']['0.25']['short_exact'] for r in rs],*[r['scores']['0.5']['merge_candidates'] for r in rs]])
    parts.append('<h2>Frozen validation</h2>'+table(['Page','Base exact','Candidate exact','Base short .25','Candidate short .25','Base merges','Candidate merges'],values))
    summary=[]
    for mode in [0,1]:
        t=totals(val,mode);r=totals(val,mode,'0.25')
        summary.append(['Baseline' if mode==0 else 'Challenger',t['truth'],t['exact'],t['merge_candidates'],r['short_exact']])
    parts.append(table(['Validation total','Truth','Strict exact','Merge boxes','Short exact .25'],summary))
    if (out/'native_confirmation.json').exists():
        native=read(out/'native_confirmation.json')
        parts.append('<h2>Full native pipeline confirmation</h2>'+table(['Page','Baseline exact','Challenger exact','Baseline agrees with probe','Challenger agrees with probe'],[[p['page'],p['base']['scores']['0.5']['exact'],p['candidate']['scores']['0.5']['exact'],p['base']['text_matches_probe'],p['candidate']['text_matches_probe']] for p in native]))
    parts.append('<p><a href="selection.json">Selection</a> · <a href="development/detection_scores.json">All cached-map settings</a> · <a href="validation/recognition.json">Validation predictions</a></p>')
    for page in read(out/'development/manifest.json'):
        for setting in [selection['baseline'],selection['winner']]:
            row=next(r for r in dev if r['page']==page['id'] and r['setting']==setting)
            svg=overlay(page,row['words']).replace('../../../output/pdf/quality/'+page['image'],os.path.relpath(page['image'],out).replace(os.sep,'/'))
            parts.append(f'<details><summary>{page["id"]}: setting {setting}, green truth / red output</summary>{svg}</details>')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8');print(out/'report.html')


def confirm(out):
    selection=read(out/'selection.json');pages=read(out/'development/manifest.json')+read(out/'validation/manifest.json')
    workload=out/'native_workload.json';write(workload,dict(pages=[dict(id=p['id'],image=p['image']) for p in pages]))
    expected={}
    for split,indices in [('development',[selection['baseline'],selection['winner']]),('validation',[0,1])]:
        for row in read(out/split/'recognition.json'):
            if row['setting'] in indices:expected[(row['page'],'base' if row['setting']==indices[0] else 'candidate')]=row
    confirmations={p['id']:dict(page=p['id']) for p in pages}
    for mode,params in [('base',BASE),('candidate',selection['candidate'])]:
        folder=out/f'native_{mode}'
        subprocess.run([str(ROOT/'target/release/throughput.exe'),'--workload',str(workload),'--output',str(folder),'--pages',str(len(pages)),
                        '--size','1536','--reco-batch','256','--workers','2','--inflight','3','--arena-mib','6144','--det-arena-mib','4096',
                        '--page-orientation','--deskew','--dense-refine','--dense-bin-thresh',str(params['bin_thresh']),
                        '--dense-box-thresh',str(params['box_thresh']),'--dense-unclip-ratio',str(params['unclip_ratio'])],cwd=ROOT,env=environment(),check=True,timeout=1200)
        native=read(folder/'summary.json')
        for i,page in enumerate(pages):
            record=native['first_pages'][str(i)];words=[dict(w,polygon=w.get('quadrilateral',w['polygon'])) for w in record['words']]
            probe=expected[(page['id'],mode)]
            confirmations[page['id']][mode]=dict(scores=scores(page['words'],words),text_matches_probe=[w['text'] for w in words]==[w['text'] for w in probe['words']],
                                               geometry_matches_probe=bool(len(words)==len(probe['words']) and np.allclose([w['polygon'] for w in words],[w['polygon'] for w in probe['words']],atol=1e-6,rtol=0)))
    write(out/'native_confirmation.json',list(confirmations.values()))
    write(out/'native_binary_provenance.json',{'sha256':hashlib.sha256((ROOT/'target/release/throughput.exe').read_bytes()).hexdigest()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    out=(args.output or ROOT/'pybaseline/results'/('dense_tuning_'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    if args.report_only:return report(out)
    out.mkdir(parents=True,exist_ok=False)
    fixtures=ROOT/'output/pdf/quality';manifest=read(fixtures/'manifest.json');pages=[dict(p,image=str((fixtures/p['image']).resolve())) for p in manifest['pages'] if p['id'] in [f'statement_{i}_cw0' for i in range(3)]]
    validation=new_pages(out/'fixtures');write(out/'validation_manifest_frozen.json',validation)
    paths=[Path(p['image']) for p in pages+validation]+[ROOT/'models'/f for f in ['db_resnet34.onnx','parseq.onnx','metadata.json','page_orientation.onnx','page_orientation.json']]+[ROOT/'target/release/refinement_probe.exe',ROOT/'target/release/throughput.exe']
    write(out/'provenance.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    grid=[dict(bin_thresh=b,box_thresh=.1,unclip_ratio=u) for b in [.2,.3,.4,.5,.6] for u in [.75,1.,1.25,1.5,1.75]];baseline=grid.index(BASE)
    dev=out/'development';cache(dev,pages);rows=postprocess(dev,grid)
    strict=sorted(range(len(grid)),key=lambda i:(-totals(rows,i)['matched'],totals(rows,i)['predicted'],abs(grid[i]['unclip_ratio']-1.5)))
    relaxed=sorted(range(len(grid)),key=lambda i:(-totals(rows,i,'0.25')['matched'],totals(rows,i,'0.25')['merge_candidates']))
    per_threshold=[max([i for i,p in enumerate(grid) if p['bin_thresh']==b],key=lambda i:(totals(rows,i)['matched'],-totals(rows,i)['predicted'])) for b in [.2,.3,.4,.5,.6]]
    shortlist=list(dict.fromkeys([baseline,*strict[:3],*relaxed[:2],*per_threshold]));write(out/'shortlist.json',dict(baseline=baseline,shortlist=shortlist))
    print('Recognition shortlist:',[grid[i] for i in shortlist],flush=True);recognized=recognize(dev,rows,shortlist)
    # Box matches give an upper bound on exact words. Recognize every other setting
    # that could beat the measured baseline, rather than trusting box ranking alone.
    additional=[i for i in range(len(grid)) if i not in shortlist and totals(rows,i)['matched']>totals(recognized,baseline)['exact']]
    if additional:
        initial=recognized;write(dev/'recognition_initial.json',initial)
        recognized=initial+recognize(dev,rows,additional);write(dev/'recognition.json',recognized);shortlist+=additional
    selection=choose(recognized,grid,baseline,shortlist);write(out/'selection.json',selection)
    print('Frozen challenger:',selection['candidate'],flush=True)
    val=out/'validation';cache(val,validation);rows=postprocess(val,[BASE,selection['candidate']]);recognize(val,rows,[0,1]);confirm(out);report(out)

if __name__=='__main__':main()
