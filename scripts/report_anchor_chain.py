"""Contact sheet and evaluation for the fixed one-hop anchor experiment."""
import argparse
import html
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tune_dense_detection import read,write
from pybaseline.metrics import polygon


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'output/diagnostics/anchor-chain')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>One-hop anchor recovery</title><style>body{font:15px system-ui;color:#203247;max-width:1450px;margin:30px auto;padding:0 20px}p{line-height:1.5}h2{overflow-wrap:anywhere}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd}th{background:#e1edf5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{border:1px solid #ccd;padding:12px;min-width:0}svg{width:100%;height:160px}</style><h1>One-hop anchor recovery</h1><p>Diagnostic experiment: a discarded thin component must have exactly one direct text anchor, which in turn must have a second aligned word beyond it. One hop only. Detector thresholds, crop limits and recognition confidence gate are unchanged. Existing production recovery is the baseline. No language correction and no production change.</p><p>The original 17 variants informed this rule; new holdout layouts were generated after freezing it. Rotated copies repeat content and are not independent documents. Green = correct text with at least half of exactly one reference word covered; red = another addition, including a box with insufficient overlap. Counts of no-overlap omissions are separate from correct recoveries.</p>']
    gallery=[];summaries=[]
    for folder in args.results:
        rows=read(folder/'results.json');pages={p['id']:p for p in read(folder/'manifest.json')};totals=[0]*7
        parts.append('<h2>'+html.escape(str(folder))+'</h2><div class="scroll"><table><tr><th>Page</th><th>Proposed</th><th>Accepted</th><th>Correct overlap/text</th><th>Other additions</th><th>No overlap before</th><th>After</th><th>Strict exact gain</th></tr>')
        for row in rows:
            correct=sum(a['correct'] for a in row['additions'])
            vals=[row['candidates'],len(row['accepted']),correct,len(row['accepted'])-correct,row['before'].get('no_box_overlap',0),row['after'].get('no_box_overlap',0),row['after'].get('exact',0)-row['before'].get('exact',0)]
            totals=[a+b for a,b in zip(totals,vals)]
            parts.append('<tr><td>'+row['page']+'</td>'+''.join('<td>'+str(v)+'</td>' for v in vals)+'</tr>')
            for word,addition in zip(row['accepted'],row['additions']):gallery.append((pages[row['page']],word,addition))
        parts.append('<tr><th>Total</th>'+''.join('<th>'+str(v)+'</th>' for v in totals)+'</tr></table></div>')
        summaries.append(dict(source=str(folder),totals=dict(zip(['proposed','accepted','correct','other','missing_before','missing_after','strict_gain'],totals))))
    parts.append('<h2>Accepted additions</h2><p>Blue outlines: nearby reference words. Coloured outline: added prediction. Original source pixels, including skew/rotation.</p><div class="cards">')
    for page,word,addition in gallery:
        W,H=page['width'],page['height'];shape=polygon(word['polygon']);x0,y0,x1,y1=shape.bounds
        x=max(0,x0*W-110);y=max(0,y0*H-35);cw=min(W-x,(x1-x0)*W+220);ch=min(H-y,(y1-y0)*H+70)
        src=html.escape(os.path.relpath(page['image'],out).replace(os.sep,'/'),quote=True)
        points=' '.join(f'{a*W},{b*H}' for a,b in word['polygon']);color='#12804a' if addition['correct'] else '#ce2539'
        refs=''
        for t in page['words']:
            if polygon(t['polygon']).intersects(shape):
                pts=' '.join(f'{a*W},{b*H}' for a,b in t['polygon']);refs+=f'<polygon points="{pts}" fill="none" stroke="#007baf" stroke-width="1"/>'
        parts.append(f'<div class="card"><b>{html.escape(word["text"])}</b> ({word["confidence"]:.3f}) / {page["id"]}<br>Reference with ≥50% coverage: {html.escape(str(addition["truth"]))}<svg viewBox="{x} {y} {cw} {ch}"><image href="{src}" width="{W}" height="{H}"/>{refs}<polygon points="{points}" fill="none" stroke="{color}" stroke-width="2"/></svg></div>')
    parts.append('</div>');(out/'index.html').write_text(''.join(parts),encoding='utf-8');write(out/'summary.json',summaries);print(summaries);print(out/'index.html')


if __name__=='__main__':main()
