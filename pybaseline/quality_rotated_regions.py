"""Targeted, image-triggered rotated-region rescans; bounded accuracy experiment."""
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from doctr.models import crop_orientation_predictor
from pybaseline.quality_candidates import candidates
from pybaseline.quality_study import Engine,rotate,remap
from pybaseline.quality_experiments import analyze
from pybaseline.metrics import polygon

def crop(image,poly,angle):
    h,w=image.shape[:2];a=math.radians(angle);basis=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]);points=np.array(poly)*[w,h]@basis;lo=points.min(axis=0);hi=points.max(axis=0)
    source=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])@basis.T;cw=max(1,int(round(hi[0]-lo[0])));ch=max(1,int(round(hi[1]-lo[1])))
    dest=np.array([[0,0],[cw,0],[cw,ch],[0,ch]],np.float32);matrix=cv2.getPerspectiveTransform(source.astype(np.float32),dest)
    return cv2.warpPerspective(image,matrix,(cw,ch),borderValue=(255,255,255)),matrix

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-inference',action='store_true');p.add_argument('--size',type=int,default=512);args=p.parse_args()
    if not args.run_inference:p.error('Explicit --run-inference required')
    out=Path('pybaseline/results/quality');root=Path('output/pdf/quality');manifest=json.loads((root/'manifest.json').read_text());engine=Engine();orientation=crop_orientation_predictor(pretrained=True).eval().cpu();rows=[]
    for page in [p for p in manifest['pages'] if p['id'] in ['statement_0_cw0','statement_1_cw0','statement_2_cw0']]:
        image=np.array(Image.open(root/page['image']).convert('RGB'));h,w=image.shape[:2];base=json.loads((out/f"{page['id']}_size1536.json").read_text())['words'];regions=candidates(image,base);result=list(base);details=[]
        for region in regions:
            # Include intersecting existing boxes so a partial word cannot be cut at the ROI edge.
            region_poly=polygon(region['polygon']);extra=[]
            for word in base:
                geom=polygon(word['polygon'])
                if geom.intersection(region_poly).area>0:extra.extend(list(geom.exterior.coords)[:-1])
            if extra:
                points=np.r_[np.array(region['polygon']),np.array(extra)]*[w,h]
                angle_rad=math.radians(region['angle']);basis=np.array([[math.cos(angle_rad),-math.sin(angle_rad)],[math.sin(angle_rad),math.cos(angle_rad)]])
                aligned=points@basis;lo=aligned.min(axis=0)-8;hi=aligned.max(axis=0)+8
                region['polygon']=((np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])@basis.T)/[w,h]).tolist()
            roi,matrix=crop(image,region['polygon'],region['angle']);angle=orientation([roi])[1][0]
            if angle not in (0,180):angle=0 # geometry already fixes the long-axis direction
            fixed,rotation=rotate(roi,angle);predicted=remap(engine.run(fixed,args.size),rotation,roi.shape,fixed.shape);inv=np.linalg.inv(matrix);converted=[]
            for word in predicted:
                points=np.array(word['polygon'])*[roi.shape[1],roi.shape[0]];xy=cv2.perspectiveTransform(points[None].astype(np.float64),inv)[0]/[w,h]
                converted.append(dict(word,polygon=xy.tolist()))
            region_poly=polygon(region['polygon'])
            accepted=sum(len(word['text']) for word in converted if word['confidence']>=.5)>=3
            if accepted:
                result=[word for word in result if not region_poly.contains(polygon(word['polygon']).centroid)]
                result.extend(converted)
            details.append(dict(region=region,flip=angle,accepted=accepted,words=converted))
        row=dict(page=page['id'],roi_size=args.size,regions=details,words=result,**analyze(page['words'],result));rows.append(row)
        print(page['id'],'regions',len(regions),'exact',row['scores']['0.5']['exact'],'local90',row['regions_iou25']['local90'],'local45',row['regions_iou25']['local45'],flush=True)
        (out/f"rescanned_{page['id']}.json").write_text(json.dumps(row,indent=2))
    (out/'rotated_roi_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
