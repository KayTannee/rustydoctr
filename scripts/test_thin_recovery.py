"""Experimental, bounded thin-component recovery using cached native detector maps."""
import argparse
from datetime import datetime
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.tune_dense_detection import read,write,recognize,map_words
from scripts.audit_missing_words import analyse
from pybaseline.metrics import polygon


def candidates(prob):
    size=prob.shape[0]
    mask=(prob>=.3).astype(np.uint8)
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    _,_,anchors,_=cv2.connectedComponentsWithStats(opened,8)
    anchors=[tuple(map(float,a[:4])) for a in anchors[1:] if a[2]>=3 and a[3]>=4 and a[2]>=a[3]]
    result=[]
    for index,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h<3 or h>60 or w>h*.6 or area<3 or area/(w*h)<.25:
            continue
        if opened[y:y+h,x:x+w][labels[y:y+h,x:x+w]==index].any():
            continue
        peers=[]
        for ax,ay,aw,ah in anchors:
            gap=max(ax-(x+w),x-(ax+aw),0)
            if .4<=h/ah<=2.0 and abs(y+h/2-ay-ah/2)<=.35*max(h,ah) and gap<=5*max(h,ah):
                peers.append((gap,ax,ay,aw,ah))
        if len(peers)<2:
            continue
        peers=sorted(peers)[:2]
        # The DB interior is smaller than the glyph; add height-relative context.
        line_top=float(np.median([p[2] for p in peers]))
        line_bottom=float(np.median([p[2]+p[4] for p in peers]))
        pad=.3*max(h,line_bottom-line_top)
        box=[max(0,x-pad),max(0,min(y,line_top)-pad),min(size,x+w+pad),min(size,max(y+h,line_bottom)+pad)]
        score=float(prob[y:y+h,x:x+w][labels[y:y+h,x:x+w]==index].mean())
        if score<.3:
            continue
        result.append(dict(box=box,objectness=score,raw=[int(x),int(y),int(w),int(h)],peers=peers))
    return sorted(result,key=lambda r:-r['objectness'])[:32]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'pybaseline/results/dense_tuning_v2')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--recognize',action='store_true')
    parser.add_argument('--single',action='store_true',help='Source is one cache/manifest/recognition folder, baseline setting 0')
    args=parser.parse_args();out=args.output or ROOT/"pybaseline/results"/("thin_recovery_"+datetime.now().strftime("%Y%m%d_%H%M%S"));out.mkdir(parents=True,exist_ok=False)
    write(out/"experiment.json",dict(source=str(args.source.resolve()),script=Path(__file__).read_text(encoding="utf-8"),candidate_cap=32,confidence_gate=.9))
    all_results=[]
    for split,setting in ([('',0)] if args.single else [('development',8),('validation',0)]):
        source=args.source/split;folder=out/split;folder.mkdir(exist_ok=True)
        pages=read(source/'manifest.json');infos=read(source/'cache/cache.json')
        baseline={r['page']:r for r in read(source/'recognition.json') if r['setting']==setting}
        write(folder/'manifest.json',pages)
        # The recognizer consumes this cache layout. Copy only the small metadata/raster set.
        (folder/'cache').mkdir(exist_ok=True)
        shutil.copy2(source/'cache/cache.json',folder/'cache/cache.json')
        for info in infos:shutil.copy2(source/'cache'/info['image'],folder/'cache'/info['image'])
        rows=[]
        for page,info in zip(pages,infos):
            assert page['id']==info['id']
            extras=[]
            maps=[(tile,file,1024) for tile,file in info['tiles']]
            if info.get('base_map'):
                maps.append((dict(x=0,y=0,width=info['width'],height=info['height'],owner_start=0,owner_end=info['width']),info['base_map'],1536))
            for tile,file,size in maps:
                prob=np.fromfile(source/'cache'/file,dtype='<f4').reshape(size,size)
                for candidate in candidates(prob):
                    b=np.array(candidate['box']).reshape(2,2)/size
                    tw,th=tile['width'],tile['height']
                    if tw>th:b[:,1]=(b[:,1]-.5)*tw/th+.5
                    else:b[:,0]=(b[:,0]-.5)*th/tw+.5
                    b=(b*[tw,th]+[tile['x'],tile['y']])/[info['width'],info['height']]
                    cx=b[:,0].mean()*info['width']
                    if not tile['owner_start']<=cx<tile['owner_end']:continue
                    word=dict(polygon=b.tolist(),objectness=candidate['objectness'],text='',confidence=0.,evidence=candidate)
                    mapped=map_words([word],info,page)[0];shape=polygon(mapped['polygon'])
                    # Add only boxes wholly separate from established detections.
                    if any(shape.intersection(polygon(w['polygon'])).area>0 for w in baseline[page['id']]['words']):continue
                    if any(polygon(word['polygon']).intersects(polygon(w['polygon'])) for w in extras):continue
                    extras.append(word)
            bounded=[];pixels=0
            for word in sorted(extras,key=lambda w:-w['objectness']):
                b=np.array(word['polygon']);area=int(np.prod((b[1]-b[0])*[info['width'],info['height']]))
                if len(bounded)>=32:break
                if pixels+area>1_000_000:continue
                bounded.append(word);pixels+=area
            extras=bounded
            rows.append(dict(page=page['id'],setting=0,params={},words=extras))
            print(page['id'],len(extras),'candidates',flush=True)
        write(folder/'candidates.json',rows)
        if args.recognize:
            recognized=recognize(folder,rows,[0])
            for page,row in zip(pages,recognized):
                accepted=[w for w in row['words'] if w['confidence']>=.9 and 1<=len(w['text'])<=2 and w['text'].isalnum()]
                before=baseline[page['id']]['words'];after=before+accepted
                details=analyse(page,after);previous=analyse(page,before)
                gained=[dict(before=a,after=b) for a,b in zip(previous,details) if a['status']!=b['status']]
                additions=[]
                for word in accepted:
                    shape=polygon(word['polygon'])
                    hits=[i for i,w in enumerate(page['words']) if polygon(w['polygon']).area and shape.intersection(polygon(w['polygon'])).area/polygon(w['polygon']).area>=.5]
                    additions.append(dict(text=word['text'],confidence=word['confidence'],truth_indices=hits,
                                          truth=[page['words'][i]['text'] for i in hits],
                                          correct=len(hits)==1 and page['words'][hits[0]]['text']==word['text']))
                result=dict(page=page['id'],candidates=len(row['words']),accepted=accepted,
                            additions=additions,
                            recognition=[dict(text=w['text'],confidence=w['confidence']) for w in row['words']],
                            before=dict(Counter(w['status'] for w in previous if w['group']=='ordinary')),
                            after=dict(Counter(w['status'] for w in details if w['group']=='ordinary')),changes=gained)
                all_results.append(result)
                print(page['id'],len(accepted),'accepted',result['recognition'],flush=True)
    if args.recognize:write(out/'results.json',all_results)


if __name__=='__main__':main()
