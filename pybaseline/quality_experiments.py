"""Accuracy-only rotation and selective-region experiments. Not production defaults."""
import argparse,json,math
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from doctr.models import page_orientation_predictor,crop_orientation_predictor
from doctr.models._utils import estimate_orientation
from pybaseline.quality_study import Engine,rotate,skew,remap,offsets
from pybaseline.metrics import score_words,detection_summary

def analyze(truth,words):
    pairs=dict(score_words(truth,words,.25,True)['pairs'])
    singles=[dict(truth=w['text'],predicted=words[pairs[i]]['text'] if i in pairs else '<unmatched>',region=w['region']) for i,w in enumerate(truth) if len(w['text'])==1]
    regions={}
    for region in sorted({w['region'] for w in truth}):
        ids=[i for i,w in enumerate(truth) if w['region']==region]
        regions[region]=dict(truth=len(ids),matched=sum(i in pairs for i in ids),exact=sum(i in pairs and truth[i]['text']==words[pairs[i]]['text'] for i in ids))
    return dict(scores={str(iou):detection_summary([score_words(truth,words,iou)]) for iou in [.25,.5]},regions_iou25=regions,single_characters=singles)

def dense_band(image,size=1536):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    _,labels,stats,_=cv2.connectedComponentsWithStats((gray<160).astype(np.uint8),8)
    del labels
    # Text-sized components only; long rules and huge dark title blocks are excluded.
    useful=[s for s in stats[1:] if 3<=s[4] and 3<=s[3]<=h*.04 and s[2]<=w*.04 and s[3]<=s[2]*12]
    bins={};step=64
    for x,y,cw,ch,area in useful:
        key=(y+ch//2)//step;group=bins.setdefault(key,[]);group.append((x,y,cw,ch,area))
    good=sorted(k for k,v in bins.items() if len(v)>=20 and np.median([s[3] for s in v])*size/max(h,w)<9)
    groups=[]
    for key in good:
        if groups and key<=groups[-1][-1]+1:groups[-1].append(key)
        else:groups.append([key])
    if not groups:return None
    group=max(groups,key=lambda g:sum(len(bins[k]) for k in g));components=[s for k in group for s in bins[k]]
    y0=max(0,min(s[1] for s in components)-24);y1=min(h,max(s[1]+s[3] for s in components)+24)
    return dict(y0=int(y0),y1=int(y1),components=len(components),median_height_pixels=float(np.median([s[3] for s in components])),reason='dense small image components; no truth labels used')

def refine(engine,image,base):
    band=dense_band(image)
    if band is None:return base,dict(regions=[],detector_pixels=1536**2)
    h,w=image.shape[:2];y0,y1=band['y0'],band['y1'];mid=w//2;overlap=int(w*.04)
    result=[word for word in base if not y0/h <=np.array(word['polygon'])[:,1].mean()<=y1/h]
    rois=[]
    for x0,x1,lo,hi in [(0,mid+overlap,0,mid),(mid-overlap,w,mid,w)]:
        crop=image[y0:y1,x0:x1];words=offsets(engine.run(crop,1024),(x0,y0,x1-x0,y1-y0),image.shape)
        kept=[word for word in words if lo/w<=np.array(word['polygon'])[:,0].mean()<hi/w]
        result.extend(kept);rois.append(dict(box=[x0,y0,x1,y1],kept=len(kept)))
    return result,dict(band=band,regions=rois,detector_pixels=1536**2+2*1024**2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-inference',action='store_true');a=p.parse_args()
    if not a.run_inference:p.error('Explicit --run-inference required')
    root=Path('output/pdf/quality');out=Path('pybaseline/results/quality');m=json.loads((root/'manifest.json').read_text());engine=Engine()
    page_model=page_orientation_predictor(pretrained=True).eval().cpu();results=[]
    # One pass per condition, no throughput repetitions. Shared GPU invalidates timing comparisons.
    cases=[p for p in m['pages'] if p['family']=='statement_0']
    for page in cases:
        image=np.array(Image.open(root/page['image']).convert('RGB'));quarter=page_model([image])[1][0]
        canonical=np.rot90(image,int(quarter//90)).copy();angle,votes=skew(canonical)
        old=int(estimate_orientation(canonical))
        variants=[('uncorrected',0),('rounded',quarter+round(angle)),('fractional',quarter+angle)]
        predictions={}
        for mode,correction in variants:
            if mode=='uncorrected' and page['page_rotation_deg']==0:
                words=json.loads((out/'statement_0_cw0_size1536.json').read_text())['words']
            else:
                fixed,matrix=rotate(image,correction);words=remap(engine.run(fixed,1536),matrix,image.shape,fixed.shape)
            predictions[mode]=dict(words=words,**analyze(page['words'],words))
        row=dict(page=page['id'],truth_cw=page['page_rotation_deg'],page_quarter=quarter,fine_skew=angle,line_votes=votes,doctr_raw_image_skew=old,variants=predictions)
        (out/f"rotation_{page['id']}.json").write_text(json.dumps(row,indent=2));results.append(row)
        print(page['id'],'skew',round(angle,3),'exact',[predictions[k]['scores']['0.5']['exact'] for k in predictions],flush=True)
    (out/'rotation_results.json').write_text(json.dumps(results,indent=2))
    rows=[]
    for page in [p for p in m['pages'] if p['id'] in ['statement_0_cw0','statement_1_cw0','statement_2_cw0','text_scale']]:
        image=np.array(Image.open(root/page['image']).convert('RGB'));base=json.loads((out/f"{page['id']}_size1536.json").read_text())['words']
        words,policy=refine(engine,image,base);row=dict(page=page['id'],policy=policy,words=words,**analyze(page['words'],words));rows.append(row)
        print('refine',page['id'],row['scores']['0.5']['exact'],policy,flush=True)
    (out/'refinement_results.json').write_text(json.dumps(rows,indent=2))
    # Crop diagnostic: ground truth is deliberately used ONLY to isolate recognizer/classifier behaviour.
    crop_model=crop_orientation_predictor(pretrained=True).eval().cpu();crop_rows=[]
    page=m['pages'][0];image=np.array(Image.open(root/page['image']).convert('RGB'));h,w=image.shape[:2]
    targets=[g for g in page['words'] if g['text'] in ['I','a','-','_','—'] and g['local_rotation_deg']==0]
    boxes=np.array([[*np.min(g['polygon'],axis=0),*np.max(g['polygon'],axis=0),1.] for g in targets],dtype=np.float32)
    crops=[image[max(0,int(b[1]*h)):min(h,int(b[3]*h)+1),max(0,int(b[0]*w)):min(w,int(b[2]*w)+1)] for b in boxes]
    orientation=crop_model(crops);preds=engine.recognize(image,boxes)
    for g,pred,angle,confidence in zip(targets,preds,orientation[1],orientation[2]):
        crop_rows.append(dict(truth=g['text'],region=g['region'],upright_recognized=pred['text'],crop_orientation=angle,orientation_confidence=confidence))
    (out/'isolated_characters.json').write_text(json.dumps(crop_rows,indent=2))
    print('isolated characters',len(crop_rows),'nonzero orientation',sum(r['crop_orientation']!=0 for r in crop_rows),flush=True)
if __name__=='__main__':main()
