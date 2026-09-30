"""Reproduce the official HierText sample detection scores without its Beam CLI."""
import gzip,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_public import HIER_SOURCE,hier_input,hier_evaluator
import numpy as np

def main():
    with gzip.open(HIER_SOURCE/'gt/validation.jsonl.gz','rt',encoding='utf-8') as f:gt=json.load(f)['annotations']
    with gzip.open(HIER_SOURCE/'sample_output.jsonl.gz','rt',encoding='utf-8') as f:pred={p['image_id']:p for p in json.load(f)['annotations']}
    evaluator=hier_evaluator();total=Counter()
    for index,page in enumerate(gt):
        W,H=page['image_width'],page['image_height']
        convert=lambda w:dict(text=w.get('text',''),polygon=(np.array(w['vertices'])/[W,H]).tolist(),ignore=not w.get('legible',True))
        words=[convert(w) for p in page['paragraphs'] for l in p['lines'] for w in l['words']]
        guesses=[convert(w) for p in pred.get(page['image_id'],{}).get('paragraphs',[]) for l in p['lines'] for w in l['words']]
        total.update(evaluator.evaluate_one_image(hier_input(dict(width=W,height=H,words=words),guesses)))
        if (index+1)%200==0:print('Official sample',index+1,'/',len(gt),flush=True)
    actual=evaluator.evaluate(total);expected={}
    for line in (HIER_SOURCE/'sample_eval_scores.txt').read_text(encoding='utf-8').splitlines()[1:]:
        if not line.strip():break
        key,value=line.split(':');expected[key]=float(value)
    for key,value in expected.items():assert abs(actual[key]-value)<1e-10,(key,actual[key],value)
    out=ROOT/'pybaseline/results/public_v1/evaluator_validation.json';out.write_text(json.dumps(dict(pages=len(gt),expected=expected,actual={k:actual[k] for k in expected},passed=True),indent=2),encoding='utf-8')
    print('PASS: reproduced all official sample word-detection scores on',len(gt),'images',flush=True)
if __name__=='__main__':main()
