"""Render saved thin-recovery experiments without running inference."""
import html
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tune_dense_detection import read
from pybaseline.metrics import polygon


def main():
    folders=[Path(p).resolve() for p in sys.argv[1:]]
    out=ROOT/'output/diagnostics/thin-recovery';out.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Thin-character recovery experiment</title><style>body{font:15px system-ui;color:#203247;max-width:1450px;margin:30px auto;padding:0 20px}p{line-height:1.5}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd}th{background:#e1edf5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{border:1px solid #ccd;padding:10px}svg{width:100%;height:160px}</style><h1>Thin-character recovery experiment</h1><p>Prototype using cached native DB detector maps and the existing PARSeq recognizer. Existing boxes/text are unchanged; only separate discarded thin components are proposed. Two nearby text components, compatible line geometry, at most 32 crops / one million source pixels per page, and recognition confidence ≥0.9 with one or two alphanumeric characters. No language correction.</p><p>Development and previously inspected dense pages informed the rule. Fresh numeric fixtures were generated afterwards and evaluated with the fixed rule. Rotated copies are repeated content, not independent documents. This diagnostic Python prototype is not enabled in the Rust production pipeline; these runs do not establish throughput.</p>']
    gallery=[]
    for folder in folders:
        rows=read(folder/'results.json');pages={}
        for manifest in [folder/'manifest.json',folder/'development/manifest.json',folder/'validation/manifest.json']:
            if manifest.exists():pages.update({p['id']:p for p in read(manifest)})
        parts.append('<h2>'+html.escape(folder.name)+'</h2><div class="scroll"><table><tr><th>Page</th><th>Candidates</th><th>Accepted</th><th>Correct additions*</th><th>Other additions*</th><th>Ordinary no overlap: before → after</th><th>Strict exact gain</th></tr>')
        total=[0]*7
        for r in rows:
            page=pages[r['page']];correct=0
            for word in r['accepted']:
                shape=polygon(word['polygon']);hits=[w for w in page['words'] if polygon(w['polygon']).area and shape.intersection(polygon(w['polygon'])).area/polygon(w['polygon']).area>=.5]
                good=len(hits)==1 and hits[0]['text']==word['text'];correct+=good
                gallery.append((page,word,good,[w['text'] for w in hits]))
            values=[r['candidates'],len(r['accepted']),correct,len(r['accepted'])-correct,r['before'].get('no_box_overlap',0),r['after'].get('no_box_overlap',0),r['after'].get('exact',0)-r['before'].get('exact',0)]
            total=[a+b for a,b in zip(total,values)]
            cells=[r['page'],*values[:4],f'{values[4]} → {values[5]}',values[6]]
            parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>')
        cells=['Total',*total[:4],f'{total[4]} → {total[5]}',total[6]]
        parts.append('<tr>'+''.join('<th>'+str(v)+'</th>' for v in cells)+'</tr></table></div>')
    parts.append('<p>*Correct addition: exact text and at least half of exactly one labelled word covered. This is separate from strict IoU 0.5 correctness. Other additions include wrong text, unlabelled marks, multiple-word coverage and insufficient coverage. Ordinary omission counts exclude explicit stress sequences and pure punctuation.</p><h2>Added crops</h2><p>Green: correctly recovered text under the coverage check. Red: other addition requiring review. These are original image coordinates.</p><div class="cards">')
    for page,word,good,truth in gallery[:90]:
        W,H=page['width'],page['height'];shape=polygon(word['polygon']);x0,y0,x1,y1=shape.bounds
        x=max(0,x0*W-100);y=max(0,y0*H-35);cw=min(W-x,(x1-x0)*W+200);ch=min(H-y,(y1-y0)*H+70)
        src=html.escape(os.path.relpath(page['image'],out).replace(os.sep,'/'),quote=True)
        points=' '.join(f'{a*W},{b*H}' for a,b in word['polygon']);color='#12804a' if good else '#cf2737'
        parts.append(f'<div class="card"><b>{html.escape(word["text"])}</b> ({word["confidence"]:.3f}) · {html.escape(page["id"])}<br>Reference: {html.escape(str(truth))}<svg viewBox="{x} {y} {cw} {ch}"><image href="{src}" width="{W}" height="{H}"/><polygon points="{points}" fill="none" stroke="{color}" stroke-width="2"/></svg></div>')
    parts.append('</div>')
    (out/'index.html').write_text(''.join(parts),encoding='utf-8');print(out/'index.html')


if __name__=='__main__':main()
