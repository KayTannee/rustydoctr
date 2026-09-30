"""Trace remaining ordinary-word omissions through cached native detector stages."""
import argparse
from collections import Counter
import html
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.tune_dense_detection import read,write,map_words,BASE,call
from scripts.audit_missing_words import analyse
from scripts.test_thin_recovery import candidates
from pybaseline.metrics import polygon


def maps_for(info,cache):
    maps=[(dict(x=0,y=0,width=info['width'],height=info['height'],owner_start=0,owner_end=info['width']),info['base_map'],1536,'full_page')]
    maps += [(t,f,1024,f'tile_{i}') for i,(t,f) in enumerate(info['tiles'])]
    for tile,file,size,name in maps:
        prob=np.fromfile(cache/file,dtype='<f4').reshape(size,size)
        mask=(prob>=.3).astype(np.uint8)
        opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        _,_,anchor_stats,_=cv2.connectedComponentsWithStats(opened,8)
        anchors=[a[:4].astype(float) for a in anchor_stats[1:] if a[2]>=3 and a[3]>=4 and a[2]>=a[3]]
        yield dict(tile=tile,name=name,size=size,prob=prob,mask=mask,opened=opened,labels=labels,stats=stats,anchors=anchors,proposals=candidates(prob))


def word_in_map(word,page,info,m):
    matrix=np.vstack([info['geometry']['corrected_to_original'],[0,0,1]])
    pts=np.array(word['polygon'])*[page['width'],page['height']]
    pts=(np.c_[pts,np.ones(len(pts))]@np.linalg.inv(matrix).T)[:,:2]
    t=m['tile'];cx,cy=pts.mean(axis=0)
    if not (t['owner_start']<=cx<t['owner_end'] and t['y']<=cy<t['y']+t['height']):return None
    size=m['size'];tw,th=t['width'],t['height']
    rw,rh=(size,int(size*th/tw)) if tw>=th else (int(size*tw/th),size)
    pts=(pts-[t['x'],t['y']])*[rw/tw,rh/th]+[(size-rw+1)//2,(size-rh+1)//2]
    window=np.zeros_like(m['mask']);cv2.fillConvexPoly(window,np.round(pts).astype(np.int32),1)
    return window.astype(bool)


def map_box(box,info,m,page):
    t=m['tile'];b=np.array(box,dtype=float).reshape(2,2)/m['size'];tw,th=t['width'],t['height']
    if tw>th:b[:,1]=(b[:,1]-.5)*tw/th+.5
    else:b[:,0]=(b[:,0]-.5)*th/tw+.5
    b=(b*[tw,th]+[t['x'],t['y']])/[info['width'],info['height']]
    return map_words([dict(polygon=b.tolist(),text='',confidence=0)],info,page)[0]['polygon']


def trace(word,page,info,m,final):
    window=word_in_map(word,page,info,m)
    if window is None or not window.any():return None
    ids=sorted(int(i) for i in set(m['labels'][window]) if i)
    components=[]
    for index in ids:
        x,y,w,h,area=map(int,m['stats'][index]);region=m['labels'][y:y+h,x:x+w]==index
        survives=bool(m['opened'][y:y+h,x:x+w][region].any())
        peers=[]
        for ax,ay,aw,ah in m['anchors']:
            gap=max(ax-x-w,x-ax-aw,0)
            if .4<=h/ah<=2 and abs(y+h/2-ay-ah/2)<=.35*max(h,ah) and gap<=5*max(h,ah):peers.append([float(gap),ax,ay,aw,ah])
        proposal=next((p for p in m['proposals'] if p['raw']==[x,y,w,h]),None)
        if survives:reason='survives_cleanup'
        elif h<3 or area<3:reason='tiny_or_fragmented'
        elif h>60 or w>h*.6 or area/(w*h)<.25:reason='shape_rejected'
        elif len(peers)<2:reason='insufficient_line_anchors'
        elif proposal is None:reason='score_or_candidate_cap'
        else:
            shape=polygon(map_box(proposal['box'],info,m,page))
            reason='overlaps_existing_box' if any(shape.intersection(polygon(v.get('quadrilateral',v['polygon']))).area>0 for v in final) else 'proposal_reaches_page_filter'
        components.append(dict(raw=[x,y,w,h],area=area,mean_probability=float(m['prob'][y:y+h,x:x+w][region].mean()),survives=survives,peers=peers,reason=reason))
    return dict(map=m['name'],peak=float(m['prob'][window].max()),threshold_pixels=int(m['mask'][window].sum()),opened_pixels=int(m['opened'][window].sum()),components=components)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'pybaseline/results/remaining_60_v1')
    args=parser.parse_args();out=args.results;source=out/'source';cache=source/'cache'
    pages=read(source/'manifest.json');infos={p['id']:p for p in read(cache/'cache.json')}
    final={p['id']:p['words'] for p in read(ROOT/'pybaseline/results/native_thin_v1/2_on/summary.json')['first_pages'].values()}
    write(out/'grid.json',[BASE])
    # Historical audit: preserve the pre-fix ownership/anchor behavior.
    import subprocess
    from scripts.tune_dense_detection import environment
    subprocess.run([str(ROOT/'.cache/native-thin-v1/refinement_probe.exe'), '--cache',str(cache),'--grid',str(out/'grid.json'),'--output',str(out/'pre_recognition.json'),'--thin-recovery','--dump-tiles'],env=environment(),check=True)
    pre_rows={r['page']:r for r in read(out/'pre_recognition.json')}
    pre={k:r['words'] for k,r in pre_rows.items()}
    omissions=[]
    ranking=['survives_cleanup','tiny_or_fragmented','shape_rejected','insufficient_line_anchors','score_or_candidate_cap','overlaps_existing_box','proposal_reaches_page_filter']
    for page in pages:
        info=infos[page['id']];misses=[w for w in analyse(page,final[page['id']]) if w['group']=='ordinary' and w['status']=='no_box_overlap']
        maps=list(maps_for(info,cache))
        mapped_pre=map_words(pre[page['id']],info,page)
        for word in misses:
            traces=[t for m in maps if (t:=trace(word,page,info,m,final[page['id']]))]
            reasons=[c['reason'] for t in traces for c in t['components']]
            category=max(reasons,key=ranking.index) if reasons else 'below_threshold'
            shape=polygon(word['polygon'])
            hit=[w for w in mapped_pre if polygon(w['polygon']).intersection(shape).area>0]
            if hit:category='recognition_gate' if any(w.get('thin_recovery') for w in hit) else 'pre_recognition_geometry_mismatch'
            rejected_tiles=[]
            for tile_row in pre_rows[page['id']].get('unmerged_tiles',[]):
                t=tile_row['tile']
                for candidate in tile_row['words']:
                    box=(np.array(candidate['polygon'])*[t['width'],t['height']]+[t['x'],t['y']])
                    cx,cy=box.mean(axis=0);owned=t['owner_start']<=cx<t['owner_end'] and t['y']<=cy<=t['y']+t['height']
                    candidate=dict(candidate,polygon=(box/[info['width'],info['height']]).tolist())
                    mapped=map_words([candidate],info,page)[0];coverage=shape.intersection(polygon(mapped['polygon'])).area/shape.area
                    if coverage>=.5 and not owned:rejected_tiles.append(dict(tile=t,center=[float(cx),float(cy)],coverage=coverage,polygon=mapped['polygon']))
            if len(rejected_tiles)>=2:category='tile_ownership_gap'
            omissions.append(dict(page=page['id'],image=page['image'],width=page['width'],height=page['height'],angle=page['page_rotation_deg'],word=word,traces=traces,category=category,rejected_tiles=rejected_tiles))
        print(page['id'],len(misses),'omissions',flush=True)
    assert len(omissions)==60,len(omissions)
    counts=dict(Counter(w['category'] for w in omissions))
    write(out/'audit.json',dict(counts=counts,omissions=omissions,classification='Highest reached component stage across eligible maps; diagnostic label windows may include partial components.'))
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Remaining 60 omissions</title><style>body{font:15px system-ui;color:#203247;max-width:1500px;margin:30px auto;padding:0 20px}p{line-height:1.5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:15px}.card{border:1px solid #ccd;padding:12px;min-width:0}svg{width:100%;height:145px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}</style><h1>Remaining 60 ordinary-word omissions</h1><p>Saved native recovery-on results: 17 variants of nine source layouts. Artificial character sequences and pure punctuation are excluded. Every highlighted word has no overlapping final detection. Categories trace the furthest stage reached by components intersecting the labelled window across full-page and eligible tile maps. These labels diagnose a pipeline stage; they do not prove every above-threshold pixel belongs to the glyph.</p><pre>'+html.escape(str(counts))+'</pre><p>Red: missed labelled word. All 60 occurrences are shown, including repeated rotation variants. Ground truth is used only for this audit, never to propose recovery candidates.</p><div class="cards">']
    for row in sorted(omissions,key=lambda r:(r['category'],r['page'],r['word']['index'])):
        p=row;w=row['word'];W,H=p['width'],p['height'];x0,y0,x1,y1=polygon(w['polygon']).bounds
        x=max(0,x0*W-140);y=max(0,y0*H-50);cw=min(W-x,(x1-x0)*W+280);ch=min(H-y,(y1-y0)*H+100)
        src=html.escape(os.path.relpath(p['image'],out).replace(os.sep,'/'),quote=True);points=' '.join(f'{a*W},{b*H}' for a,b in w['polygon'])
        evidence=dict(maps=p['traces'],rejected_tiles=p['rejected_tiles'])
        parts.append(f'<div class="card"><b>{html.escape(w["text"])}</b> / {p["category"]}<br>{p["page"]} / {w["region"]}<svg viewBox="{x} {y} {cw} {ch}"><image href="{src}" width="{W}" height="{H}"/><polygon points="{points}" fill="none" stroke="#ce2539" stroke-width="2"/></svg><details><summary>Stage evidence</summary><pre>{html.escape(str(evidence))}</pre></details></div>')
    parts.append('</div><p><a href="audit.json">Machine-readable evidence</a></p>')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8');print(counts);print(out/'report.html')


if __name__=='__main__':main()
