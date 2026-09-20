"""Reuse detector maps to compare word-independent versus line-supported crop orientation."""
import argparse,json,math
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
from doctr.models import crop_orientation_predictor
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.detection._utils import _remove_padding
from doctr.utils.geometry import extract_rcrops
from doctr.models._utils import rectify_crops
from doctr.models.recognition.predictor._utils import split_crops,remap_preds
from pybaseline.quality_study import Engine
from pybaseline.quality_experiments import analyze

def decode(engine,crops):
    pieces,mapping,needs=split_crops(crops,8,6,.5);pred=[]
    for batch in engine.rpre(pieces):pred.extend(engine.rpost(torch.from_numpy(engine.rec.run(None,{'input':batch.numpy()})[0])))
    return remap_preds(pred,mapping,.5) if needs else pred

def geometry(poly,shape):
    h,w=shape[:2];p=np.array(poly)*[w,h];edges=np.roll(p,-1,axis=0)-p;lengths=np.linalg.norm(edges,axis=1);idx=int(np.argmax(lengths));edge=edges[idx]
    angle=(math.degrees(math.atan2(edge[1],edge[0]))+90)%180-90
    return p.mean(axis=0),float(lengths.max()),float(lengths.min()),angle

def crop_along(image,poly,angle):
    h,w=image.shape[:2];pts=np.array(poly)*[w,h];a=math.radians(angle);basis=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    canonical=pts@basis;lo=canonical.min(axis=0);hi=canonical.max(axis=0)
    source=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])@basis.T
    width=max(1,int(round(hi[0]-lo[0])));height=max(1,int(round(hi[1]-lo[1])))
    dest=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    matrix=cv2.getPerspectiveTransform(source.astype(np.float32),dest)
    return cv2.warpPerspective(image,matrix,(width,height),borderValue=(255,255,255))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-inference',action='store_true');a=parser.parse_args()
    if not a.run_inference:parser.error('Explicit --run-inference required')
    root=Path('output/pdf/quality');out=Path('pybaseline/results/quality');manifest=json.loads((root/'manifest.json').read_text());engine=Engine();classifier=crop_orientation_predictor(pretrained=True).eval().cpu();post=DBPostProcessor(assume_straight_pages=False);results=[]
    for page in [p for p in manifest['pages'] if p['id'] in ['statement_0_cw0','statement_1_cw0','statement_2_cw0']]:
        image=np.array(Image.open(root/page['image']).convert('RGB'));_,prob=engine.detect(image,1536)
        raw=post(prob[None,:,:,None])[0][0];polys=_remove_padding([image],[{'words':raw}],True,True,False)[0]['words'][:,:4]
        crops=extract_rcrops(image,polys,assume_horizontal=False);valid=[i for i,c in enumerate(crops) if c.shape[0] and c.shape[1]];polys=polys[valid];crops=[crops[i] for i in valid]
        orientation=classifier(crops);independent=rectify_crops(crops,orientation[0]);metrics=[geometry(p,image.shape) for p in polys]
        typical=float(np.median([short for center,long,short,angle in metrics if long>short*2 and abs(angle)<15]))
        anchors=[i for i,(center,long,short,angle) in enumerate(metrics) if long>=typical*3 and long>=short*2.2 and short>=typical*.5]
        anchor_crops=[crop_along(image,polys[i],metrics[i][3]) for i in anchors];anchor_pred=classifier(anchor_crops)
        directions={i:metrics[i][3]+(180 if cls==180 and conf>=.8 else 0) for i,cls,conf in zip(anchors,anchor_pred[1],anchor_pred[2])}
        chosen=[];owners=[]
        for i,(center,long,short,own_angle) in enumerate(metrics):
            candidates=[]
            for anchor in anchors:
                ac,al,ash,aa=metrics[anchor];rad=math.radians(aa);delta=center-ac;parallel=abs(delta@np.array([math.cos(rad),math.sin(rad)]));perp=abs(delta@np.array([-math.sin(rad),math.cos(rad)]))
                if perp<=ash*.9 and parallel<=al/2+typical*6:
                    candidates.append((perp+max(0,parallel-al/2)*.15,anchor))
            owner=min(candidates)[1] if candidates else None
            angle=directions[owner] if owner is not None else 0.
            chosen.append(angle);owners.append(owner)
        context=[crop_along(image,poly,angle) for poly,angle in zip(polys,chosen)]
        modes={}
        for name,source in [('independent',independent),('line_prior',context)]:
            predictions=decode(engine,source);words=[dict(polygon=poly.tolist(),text=t,confidence=float(c)) for poly,(t,c) in zip(polys,predictions)]
            modes[name]=dict(words=words,**analyze(page['words'],words))
        row=dict(page=page['id'],modes=modes,anchor_count=len(anchors),typical_height=typical,chosen_angles=chosen,owners=owners,independent_classes=orientation[1])
        results.append(row);(out/f"local_{page['id']}.json").write_text(json.dumps(row,indent=2))
        print(page['id'],{n:(v['scores']['0.5']['exact'],v['regions_iou25']['local90']['exact'],v['regions_iou25']['local45']['exact'],sum(s['truth']==s['predicted'] for s in v['single_characters'])) for n,v in modes.items()},flush=True)
    (out/'local_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
