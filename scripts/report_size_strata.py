"""Descriptive size-stratified recall on frozen development labels; no OCR tuning."""
import json,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_public_datasets import write
from scripts.report_tiling import table

def main():
    root=ROOT/'pybaseline/results/tiling_dev_v1';results=[];body=['<h1>Accuracy by word size</h1><p>Groups use the labelled box’s vertical extent after a hypothetical whole-page fit to 1024. The grouping stays fixed across modes. This is a geometric proxy, not font size: tilted/vertical text can have a large vertical extent. Scores are polygon-matched detection/exact recall, not F1; false positives are shown in the main report.</p>']
    bins=[(0,8,'<8'),(8,12,'8–12'),(12,20,'12–20'),(20,40,'20–40'),(40,float('inf'),'40+')]
    modes={'plain1024':'plain1024','plain1536':'plain1536','selective1024':'selective1024','tiles1536_o128_heatmap/weighted':'heatmap1536','tiles2048_o128_heatmap/weighted':'heatmap2048'}
    for ds in ['funsd','cord-v2','hiertext']:
        pages=json.loads((root/ds/'manifest.json').read_text(encoding='utf-8'));body.append('<h2>'+ds+'</h2>');longs=[max(p['width'],p['height']) for p in pages];body.append(f'<p>Original image long side: {min(longs)}–{max(longs)} pixels, median {np.median(longs):.0f}; {sum(v<=1024 for v in longs)}/{len(longs)} images already fit within 1024. Enlarging existing pixels adds no source detail.</p>');rows=[]
        for path,mode in modes.items():
            diag=root/ds/path/'edge_diagnostics.json'
            if not diag.exists():continue
            data=json.loads(diag.read_text(encoding='utf-8'))['pages'];groups=defaultdict(Counter)
            for page in pages:
                for word in data[page['id']]:
                    q=np.array(word['polygon'])*[page['width'],page['height']];sourceheight=np.ptp(q[:,1]);height=sourceheight*1024/max(page['width'],page['height']);group=next(label for low,high,label in bins if low<=height<high)
                    groups[group].update(truth=1,detected=word['detected'],exact=word['exact'],no_overlap=word['no_overlap'],source_under8=int(sourceheight<8))
            for _,_,group in bins:
                c=groups[group]
                if not c['truth']:continue
                result=dict(dataset=ds,mode=mode,height_at1024=group,**c);results.append(result)
                rows.append([mode,group,c['truth'],f"{100*c['detected']/c['truth']:.1f}",f"{100*c['exact']/c['truth']:.1f}",c['no_overlap'],c['source_under8']])
        body.append(table(['Mode','Height at 1024 px','Labels','Detection recall %','Exact recall %','No overlap','Source height <8 px'],rows))
    write(root/'size_strata.json',results);(root/'size_strata.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Word size accuracy</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px;color:#203247}.scroll{overflow:auto}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}p{line-height:1.5}</style>'+''.join(body),encoding='utf-8');print(root/'size_strata.html')
if __name__=='__main__':main()

