"""Public OCR scoring: docTR metrics for forms/receipts, upstream HierText evaluator."""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from scripts.tune_dense_detection import read,write
HIER_SOURCE=ROOT/'testdata/public/hiertext/source/hiertext-70b6620b2b112597d8219e11eee9773a1403827c'

def points(word):
    p=np.asarray(word.get('quadrilateral') or word['polygon'],dtype=np.float64)
    if len(p)==2:
        (x0,y0),(x1,y1)=p;p=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    return p

def hier_evaluator():
    sys.path.insert(0,str(HIER_SOURCE))
    from evaluator.evaluator import HierTextEvaluator,TextBoxRep
    return HierTextEvaluator(text_box_type=TextBoxRep.POLY,evaluate_text=True)

def hier_input(page,words):
    scale=np.array([page['width'],page['height']])
    return dict(gt_weights=np.asarray([not w.get('ignore',False) for w in page['words']],dtype=float),gt_boxes=[points(w)*scale for w in page['words']],gt_texts=np.asarray([w['text'] for w in page['words']]),detection_boxes=[points(w)*scale for w in words],pred_texts=np.asarray([w['text'] for w in words]))

def strata(page,words):
    """Diagnostic matching; separate from either official aggregate metric."""
    gt=[w for w in page['words'] if not w.get('ignore')];ignored=[Polygon(points(w)).buffer(0) for w in page['words'] if w.get('ignore')]
    ps=[Polygon(points(w)).buffer(0) for w in words];gs=[Polygon(points(w)).buffer(0) for w in gt]
    valid=[j for j,p in enumerate(ps) if not any(p.area and p.intersection(g).area/p.area>=.5 for g in ignored)]
    ps=[ps[j] for j in valid];words=[words[j] for j in valid];ious=np.zeros((len(gs),len(ps)));overlap=np.zeros(len(gs),dtype=bool)
    tree=STRtree(ps)
    for i,g in enumerate(gs):
        for j in tree.query(g):
            area=g.intersection(ps[j]).area;overlap[i]|=area>0;union=g.area+ps[j].area-area
            if union:ious[i,j]=area/union
    pairs={}
    if gs and ps:
        ii,jj=linear_sum_assignment(-ious);pairs={int(i):int(j) for i,j in zip(ii,jj) if ious[i,j]>=.5}
    groups={}
    for name,predicate in [('all',lambda t:True),('single',lambda t:len(t)==1),('one_or_two',lambda t:len(t)<=2),('digits',lambda t:t.isdigit()),('single_digit',lambda t:len(t)==1 and t.isdigit()),('I',lambda t:t=='I')]:
        ids=[i for i,w in enumerate(gt) if predicate(w['text'])];groups[name]=dict(truth=len(ids),detected=sum(i in pairs for i in ids),exact=sum(i in pairs and gt[i]['text']==words[pairs[i]]['text'] for i in ids),no_overlap=sum(not overlap[i] for i in ids))
    recovered=[]
    for j,w in enumerate(words):
        if not w.get('thin_recovery'):continue
        matches=[i for i,g in enumerate(gs) if g.area and ps[j].intersection(g).area/g.area>=.5]
        recovered.append(dict(text=w['text'],polygon=points(w).tolist(),truth=[gt[i]['text'] for i in matches],correct_coverage=len(matches)==1 and gt[matches[0]]['text']==w['text'],strict_exact=any(pj==j and gt[i]['text']==w['text'] for i,pj in pairs.items())))
    return groups,recovered

def score_run(manifest,folder):
    pages=read(manifest);dataset=pages[0]['dataset'];out=[];totals=Counter();groups={};additions=[]
    if dataset=='hiertext':metric=hier_evaluator()
    else:
        from doctr.utils.metrics import LocalizationConfusion,OCRMetric
        det=LocalizationConfusion(use_polygons=False);ocr=OCRMetric(use_polygons=False);poly_det=LocalizationConfusion(use_polygons=True);poly_ocr=OCRMetric(use_polygons=True)
    with (folder/'pages.jsonl').open(encoding='utf-8') as f:
        for index,line in enumerate(f):
            record=json.loads(line);assert index<len(pages);page=pages[index];assert record['id']==page['id'] and record['sequence']==index,(index,record['id'],page['id']);words=record['words']
            if dataset=='hiertext':
                counts=metric.evaluate_one_image(hier_input(page,words));totals.update(counts)
            else:
                g=np.asarray([points(w) for w in page['words']],dtype=np.float32).reshape(-1,4,2);p=np.asarray([points(w) for w in words],dtype=np.float32).reshape(-1,4,2)
                gb=np.concatenate((g.min(axis=1),g.max(axis=1)),axis=1);pb=np.concatenate((p.min(axis=1),p.max(axis=1)),axis=1)
                before=(det.num_gts,det.num_preds,det.matches,ocr.raw_matches)
                det.update(gb,pb);ocr.update(gb,pb,[w['text'] for w in page['words']],[w['text'] for w in words]);poly_det.update(g,p);poly_ocr.update(g,p,[w['text'] for w in page['words']],[w['text'] for w in words])
                after=(det.num_gts,det.num_preds,det.matches,ocr.raw_matches)
                counts=dict(zip(['truth','predicted','detected','exact'],[a-b for a,b in zip(after,before)]))
            diagnostic,recovered=strata(page,words)
            for name,c in diagnostic.items():
                groups.setdefault(name,Counter()).update(c)
            additions.extend(dict(page=page['id'],**w) for w in recovered);out.append(dict(page=page['id'],counts=counts,groups=diagnostic,recovered=recovered))
            if (index+1)%100==0:print(folder.name,'scored',index+1,'/',len(pages),flush=True)
    assert len(out)==len(pages)
    if dataset=='hiertext':scores=dict(protocol='Official HierText word polygons; exact text; illegible regions ignored; no polygon downsampling',metrics=metric.evaluate(totals),counts=dict(totals))
    else:
        def summarize(d,o):
            r,p,_=d.summary();er,ep,_=o.summary();f=lambda p,r:2*p*r/(p+r) if p+r else 0
            return dict(detection_recall=r,detection_precision=p,detection_f1=f(p,r),end_to_end_recall=er['raw'],end_to_end_precision=ep['raw'],end_to_end_f1=f(ep['raw'],er['raw']),caseless_recall=er['caseless'],truth=d.num_gts,predicted=d.num_preds,detected=d.matches,exact=o.raw_matches)
        scores=dict(protocol='docTR LocalizationConfusion/OCRMetric at IoU 0.5; axis envelopes for published-style comparison, polygons also reported',metrics=summarize(det,ocr),polygons=summarize(poly_det,poly_ocr))
    scores.update(dataset=dataset,pages=len(pages),groups={n:dict(c) for n,c in groups.items()},recovered=dict(total=len(additions),correct_coverage=sum(w['correct_coverage'] for w in additions),strict_exact=sum(w['strict_exact'] for w in additions)),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    write(folder/'scores.json',scores);write(folder/'page_scores.json',out);write(folder/'recovered.json',additions);print(dataset,folder.name,scores['metrics'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'pybaseline/results/public_v1');a=p.parse_args()
    for dataset in ['funsd','cord-v2','hiertext']:
        for summary in sorted((a.root/dataset).glob('*/summary.json')):
            if not (summary.parent/'scores.json').exists():score_run(ROOT/'testdata/public'/dataset/'prepared/manifest.json',summary.parent)
if __name__=='__main__':main()
