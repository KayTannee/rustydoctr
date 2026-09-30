"""One-hop text-line support for discarded thin components; diagnostic only."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.tune_dense_detection import read,write
import scripts.test_thin_recovery as runner


def chain_candidates(prob):
    size=prob.shape[0];mask=(prob>=.3).astype(np.uint8)
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    _,_,stats2,_=cv2.connectedComponentsWithStats(opened,8)
    anchors=[tuple(map(float,a[:4])) for a in stats2[1:] if a[2]>=3 and a[3]>=4 and a[2]>=a[3]]
    result=[]
    for index,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h<3 or h>60 or w>h*.6 or area<3 or area/(w*h)<.25:continue
        region=labels[y:y+h,x:x+w]==index
        if opened[y:y+h,x:x+w][region].any():continue
        direct=[]
        for a in anchors:
            ax,ay,aw,ah=a;gap=max(ax-x-w,x-ax-aw,0)
            if .4<=h/ah<=2 and abs(y+h/2-ay-ah/2)<=.35*max(h,ah) and gap<=5*max(h,ah):direct.append(a)
        # Only extend the previously rejected single-neighbour case.
        if len(direct)!=1:continue
        first=direct[0];ax,ay,aw,ah=first;cx=x+w/2;acx=ax+aw/2;chain=[]
        for second in anchors:
            bx,by,bw,bh=second;bcx=bx+bw/2;gap=max(bx-ax-aw,ax-bx-bw,0)
            if second==first or (bcx-acx)*(acx-cx)<=0:continue
            if not (bx>=ax+aw or bx+bw<=ax):continue
            if .65<=ah/bh<=1.6 and abs(ay+ah/2-by-bh/2)<=.35*max(ah,bh) and gap<=5*max(ah,bh):chain.append((gap,second))
        if not chain:continue
        second=min(chain)[1];top=(first[1]+second[1])/2;bottom=(first[1]+first[3]+second[1]+second[3])/2
        pad=.3*max(h,bottom-top);score=float(prob[y:y+h,x:x+w][region].mean())
        if score<.3:continue
        result.append(dict(box=[max(0,x-pad),max(0,min(y,top)-pad),min(size,x+w+pad),min(size,max(y+h,bottom)+pad)],objectness=score,raw=[int(x),int(y),int(w),int(h)],peers=[first,second],policy='single_neighbour_plus_one_outward_line_anchor'))
    return sorted(result,key=lambda p:-p['objectness'])[:32]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runner.candidates=chain_candidates
    sys.argv=[sys.argv[0],'--source',str(args.source),'--output',str(args.output),'--single','--recognize']
    runner.main()
    write(args.output/'chain_policy.json',dict(source=Path(__file__).read_text(encoding='utf-8'),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Additional proposals only; baseline includes existing native thin recovery. One hop; existing thresholds and confidence gate unchanged.'))


if __name__=='__main__':main()
