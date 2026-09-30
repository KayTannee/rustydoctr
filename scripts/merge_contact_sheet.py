"""CPU-only visual audit of saved merged-word boxes; does not run inference."""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pybaseline.metrics import polygon,score_words
from shapely.strtree import STRtree


def analyse(page,record):
    truth=[polygon(w['polygon']) for w in page['words']];tree=STRtree(truth);merges=[]
    words=[dict(w,polygon=w.get('quadrilateral',w['polygon'])) for w in record['words']]
    for index,word in enumerate(words):
        shape=polygon(word['polygon'])
        ids=sorted(int(j) for j in tree.query(shape) if truth[j].area and shape.intersection(truth[j]).area/truth[j].area>=.5)
        if len(ids)<2:continue
        merges.append(dict(id=len(merges)+1,prediction_index=index,predicted=word['text'],confidence=word['confidence'],
                           polygon=word['polygon'],truth_indices=ids,truth=[page['words'][j] for j in ids],
                           regions=sorted({page['words'][j]['region'] for j in ids})))
    return dict(merges=merges,strict=score_words(page['words'],words),relaxed=score_words(page['words'],words,.25))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'pybaseline/results/line_orientation_v2')
    parser.add_argument('--orientation-results',type=Path,default=ROOT/'pybaseline/results/native_orientation_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'output/diagnostics/word-merges')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.results/'manifest.json').read_text(encoding='utf-8'))
    current=json.loads((args.results/'guided/summary.json').read_text(encoding='utf-8'))
    before=json.loads((args.orientation_results/'deskew/summary.json').read_text(encoding='utf-8'))
    current={r['id']:r for r in current['first_pages'].values()};before={r['id']:r for r in before['first_pages'].values()}
    original_manifest=json.loads((ROOT/'output/pdf/quality/manifest.json').read_text(encoding='utf-8'))
    original_pages={p['id']:p for p in original_manifest['pages']}
    provenance=json.loads((args.orientation_results/'provenance.json').read_text(encoding='utf-8'))
    provenance={k.replace('\\','/'):v for k,v in provenance.items()}
    source_manifest=ROOT/'output/pdf/quality/manifest.json'
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest()==provenance['output/pdf/quality/manifest.json'],'Historical manifest changed'
    pages=[]
    for page in manifest:
        p=dict(page,image=os.path.relpath(page['image'],out).replace(os.sep,'/'))
        p['refined']=analyse(page,current[page['id']])
        if page['id'] in before:
            assert page['words']==original_pages[page['id']]['words'],'Ground truth differs between comparison runs'
            p['rotation_only']=analyse(page,before[page['id']])
        pages.append(p)
    # Diverse, deterministic examples: avoid repeating the same strings at each angle.
    examples=[];seen=set()
    for page in pages:
        if page['page_rotation_deg']!=0:continue
        selected=[]
        for merge in page['refined']['merges']:
            key=tuple(w['text'] for w in merge['truth'])
            if key in seen:continue
            selected.append(merge);seen.add(key)
        selected.sort(key=lambda m:('legal' not in m['regions'],-len(m['truth'])))
        examples.extend(dict(page=page['id'],merge=m['id']) for m in selected[:4])
    stats=[]
    for mode in ['rotation_only','refined']:
        group=[p for p in pages if p['id'].startswith('statement') and mode in p]
        merged=[m for p in group for m in p[mode]['merges']]
        stats.append(dict(mode=mode,pages=len(group),truth=sum(p[mode]['strict']['truth'] for p in group),
                          strict_exact=sum(p[mode]['strict']['exact'] for p in group),strict_matched=sum(p[mode]['strict']['matched'] for p in group),
                          merged_boxes=len(merged),covered_truth_words=sum(len(m['truth']) for m in merged),
                          merges_with_one_or_two_char_words=sum(any(len(w['text'])<=2 for w in m['truth']) for m in merged),
                          regions=dict(Counter(r for m in merged for r in m['regions']))))
    bundle=dict(pages=pages,examples=examples,statistics=stats)
    (out/'audit.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    sources=[args.results/'manifest.json',args.results/'guided/summary.json',args.orientation_results/'deskew/summary.json',source_manifest]
    (out/'provenance.json').write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    template='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Where word boxes merge</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#203247;font:15px system-ui}main{max-width:1800px;margin:auto;padding:28px}h1{margin:0;font-size:32px}h2{margin:0 0 12px;font-size:24px}p{line-height:1.5;max-width:1150px}.muted{color:#5e6d7c}.legend{display:flex;gap:22px;flex-wrap:wrap;margin:12px 0}.key:before{content:'';display:inline-block;width:22px;height:12px;margin-right:8px;border:3px solid #d92834;vertical-align:middle;background:#ffdf9077}.truth:before{border-color:#007c96;background:transparent}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:18px 0}button,select,a.button{padding:9px 13px;border:1px solid #a6b6c5;background:white;color:#203247;border-radius:6px;font:inherit;cursor:pointer}button.active{background:#183f5c;color:white}a{color:#07699a}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card,.panel{background:white;border:1px solid #ced9e2;border-radius:9px;overflow:hidden}.card header{padding:12px 14px;border-bottom:1px solid #dde5eb;min-height:67px}.card header strong{display:block}.card svg{display:block;width:100%;height:600px}.card{cursor:pointer}.card:hover{border-color:#007c96}.section{margin:30px 0;padding:22px;background:#f8fafc;border-radius:12px}.panel{padding:18px}.viewer{display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,.75fr);gap:20px}.viewer svg{width:100%;max-height:1050px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4eb;vertical-align:top}th{background:#dfeaf2}td.words{max-width:380px;overflow-wrap:anywhere}.gallery .tile{background:white;border:1px solid #ced9e2;border-radius:7px;padding:12px;min-width:0}.tile svg{display:block;width:100%;height:155px;margin:9px 0;background:#fafafa}.tile .caption{font-size:13px;line-height:1.4;overflow-wrap:anywhere;min-height:58px}.tag{color:#a41624;font-weight:650}.small{font-size:13px}#contact-sheet{padding:22px;background:#f8fafc;border-radius:12px}#contact-sheet .muted{margin-top:6px}.notice{padding:13px 17px;border-left:4px solid #007c96;background:#e3eff5}#empty{padding:24px}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}#pageView,#gallery-section{scroll-margin-top:20px}@media(max-width:900px){main{padding:14px}.cards,.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.card svg{height:400px}.viewer{grid-template-columns:1fr}}@media(max-width:550px){.cards,.gallery{grid-template-columns:1fr}.card svg{height:540px}h1{font-size:26px}.section,#contact-sheet{padding:12px}}
</style></head><body><main>
<h1>Where word boxes merge</h1>
<p>Saved Rust OCR output: DB ResNet34 + PARSeq, 1536 detector, page orientation and deskew. The main sheet includes dense refinement and the opt-in crop policy; that policy does not alter boxes. No new inference was run to create this audit.</p>
<p class="notice"><b>Red = one predicted box covering at least half the labelled area of two or more words.</b> Blue outlines show those individual reference words. These are geometric merge candidates, not a claim that every boundary is wrong. Missing boxes and ordinary recognition errors are deliberately not highlighted here.</p>
<div class="controls"><button id="representative" class="active" onclick="setAll(false)">Six distinct upright layouts</button><button id="all" onclick="setAll(true)">All 36 page variants</button><a class="button" href="#gallery-section">Inspect close-ups</a><a class="button" href="contact-sheet.png">Download page sheet</a><a class="button" href="merge-closeups.png">Download close-ups</a></div>
<section id="contact-sheet"><h2>Merged detections across the page</h2><div class="legend"><span class="key">Predicted merged box</span><span class="key truth">Individual reference words</span></div><p class="muted small">Click a page for full-size inspection. Same source layouts repeat at several angles in the full set.</p><div id="contacts" class="cards"></div></section>
<section class="section"><h2>How much is merging?</h2><div id="stats" class="scroll"></div><p class="small muted">Totals here use the same 27 statement variants (three layouts, nine angles), not 27 independent documents. Strict correctness requires exact text and a one-to-one box match at IoU 0.5. A merged box can contain many reference words; counts of boxes and words are different quantities.</p></section>
<section id="gallery-section" class="section"><div id="closeup-sheet"><h2>Representative merged-word close-ups</h2><div class="legend"><span class="key">One OCR box</span><span class="key truth">Separate reference words</span></div><p class="small muted">Read “Reference” as the labelled word sequence. OCR text may concatenate, omit or misread those words. Examples are selected for variety, not used to estimate frequency.</p><div id="examples" class="gallery"></div></div></section>
<section id="pageView" class="section"><h2>Inspect a page</h2><div class="controls"><select id="pageSelect" onchange="selectPage(this.value,false)"></select><select id="mode" onchange="renderPage()"><option value="refined">With dense refinement</option><option value="rotation_only">Before dense refinement</option></select></div><div id="detail" class="viewer"></div></section>
<p class="muted small">Coordinates are mapped back to the original source image. Local 45-degree boxes remain axis-aligned envelopes in the upright detector path, so some red diagonal boxes are geometry/crop problems as well as multi-word coverage. Source labels use font metrics on the original fixtures and glyph extents on the newer ones. <a href="audit.json">Full machine-readable evidence</a>.</p>
</main><script>const DATA=__DATA__;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ids=Object.fromEntries(DATA.pages.map(p=>[p.id,p]));let showAll=false;
function points(poly){if(poly.length===2){const[[x,y],[a,b]]=poly;return [[x,y],[a,y],[a,b],[x,b]]}return poly}
function svg(p,merges,focus=false,overview=false){
 let view=`0 0 ${p.width} ${p.height}`;
 if(focus){const xy=merges.flatMap(m=>[...points(m.polygon),...m.truth.flatMap(w=>points(w.polygon))]);let xs=xy.map(v=>v[0]*p.width),ys=xy.map(v=>v[1]*p.height);const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);let margin=Math.max(12,(maxY-minY)*.45);const x=Math.max(0,minX-margin),y=Math.max(0,minY-margin);view=`${x} ${y} ${Math.min(p.width,maxX+margin)-x} ${Math.min(p.height,maxY+margin)-y}`;}
 const poly=(shape,color,width,fill)=>`<polygon points="${points(shape).map(([x,y])=>`${x*p.width},${y*p.height}`).join(' ')}" fill="${fill}" stroke="${color}" stroke-width="${width}"/>`;
 let s=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${view}" role="img" aria-label="Merge annotations on ${esc(p.id)}"><image href="${esc(p.image)}" width="${p.width}" height="${p.height}"/>`;
 for(const m of merges){s+=poly(m.polygon,'#d92834',overview?9:focus?1.5:4,'#ffc34b44');for(const w of m.truth)s+=poly(w.polygon,'#007c96',overview?3:focus?1:2,'none');if(!focus&&!overview){const pt=points(m.polygon)[0];s+=`<text x="${pt[0]*p.width}" y="${pt[1]*p.height-6}" font-size="25" font-weight="bold" fill="#ba1524" stroke="white" stroke-width="5" paint-order="stroke">M${m.id}</text>`}}
 return s+'</svg>';
}
function setAll(all){showAll=all;document.querySelector('#all').classList.toggle('active',all);document.querySelector('#representative').classList.toggle('active',!all);renderContacts()}
function renderContacts(){document.querySelector('#contacts').innerHTML=DATA.pages.filter(p=>showAll||p.page_rotation_deg===0).map(p=>`<article class="card" onclick="selectPage('${p.id}',true)" tabindex="0" onkeydown="if(event.key==='Enter')selectPage('${p.id}',true)"><header><strong>${esc(p.id)}</strong><span class="muted small">${p.refined.merges.length} merged boxes · ${p.refined.strict.exact}/${p.refined.strict.truth} strict exact words</span></header>${svg(p,p.refined.merges,false,true)}</article>`).join('')}
function selectPage(id,scroll){document.querySelector('#pageSelect').value=id;renderPage();if(scroll)document.querySelector('#pageView').scrollIntoView({behavior:'smooth'})}
function renderPage(){const p=ids[document.querySelector('#pageSelect').value];const selector=document.querySelector('#mode');selector.options[1].disabled=!p.rotation_only;if(!p.rotation_only)selector.value='refined';const result=p[selector.value];document.querySelector('#detail').innerHTML=`<div class="panel">${svg(p,result.merges)}</div><div><p><b>${result.merges.length} merged boxes</b>; ${result.strict.exact}/${result.strict.truth} strict exact words.</p><div class="scroll"><table><thead><tr><th>Box / region</th><th>Reference words → OCR</th></tr></thead><tbody>${result.merges.map(m=>`<tr><td>M${m.id}<br>${esc(m.regions.join(', '))}</td><td class="words">${esc(m.truth.map(w=>w.text).join(' | '))}<br><b>→ ${esc(m.predicted)}</b><br><span class="muted">confidence ${(m.confidence*100).toFixed(2)}%</span>${svg(p,[m],true)}</td></tr>`).join('')}</tbody></table></div></div>`}
document.querySelector('#pageSelect').innerHTML=DATA.pages.map(p=>`<option value="${p.id}">${esc(p.id)}</option>`).join('');
document.querySelector('#stats').innerHTML=`<table><thead><tr><th>Pipeline</th><th>Strict exact / 12,006</th><th>Merged boxes</th><th>Cover a 1–2 character word</th><th>Regions</th></tr></thead><tbody>${DATA.statistics.map(s=>`<tr><td>${s.mode==='refined'?'With dense refinement':'Rotation + deskew only'}</td><td>${s.strict_exact.toLocaleString()}</td><td>${s.merged_boxes}</td><td>${s.merges_with_one_or_two_char_words} / ${s.merged_boxes}</td><td>${Object.entries(s.regions).map(([k,v])=>`${esc(k)}: ${v}`).join(', ')}</td></tr>`).join('')}</tbody></table>`;
document.querySelector('#examples').innerHTML=DATA.examples.map(e=>{const p=ids[e.page],m=p.refined.merges.find(m=>m.id===e.merge);return `<article class="tile"><b>${esc(p.id)} · M${m.id}</b><div class="small muted">${esc(m.regions.join(', '))}</div>${svg(p,[m],true)}<div class="caption"><b>Reference:</b> ${esc(m.truth.map(w=>w.text).join(' | '))}<br><b class="tag">OCR:</b> ${esc(m.predicted)} <span class="muted">(${(m.confidence*100).toFixed(2)}%)</span></div></article>`}).join('');
renderContacts();selectPage('statement_2_cw0',false);
</script></body></html>'''
    document=template.replace('__DATA__',json.dumps(bundle,ensure_ascii=True).replace('<','\\u003c'))
    (out/'index.html').write_text(document,encoding='utf-8')
    print(json.dumps(stats,indent=2));print(out/'index.html')


if __name__=='__main__':main()
