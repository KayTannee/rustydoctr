"""Small deterministic docTR oracle fixtures for native DB geometry tests (no GPU)."""
import json
from pathlib import Path
import numpy as np
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.detection._utils import _remove_padding

def main():
    rng=np.random.default_rng(17);cases=[]
    for index in range(20):
        prob=np.zeros((48,64),np.float32)
        for _ in range(6):
            x,y=rng.integers(0,60),rng.integers(0,44)
            w,h=rng.integers(3,18),rng.integers(3,12)
            prob[y:y+h,x:x+w]=rng.choice([0.25,0.5,0.9])
        # Keep model input square, as required by current unpadding semantics.
        prob=np.pad(prob,((0,16),(0,0)))
        ih,iw=[(64,64),(90,64),(64,90)][index%3]
        boxes=DBPostProcessor()(prob[None,:,:,None])[0][0]
        boxes=_remove_padding([np.empty((ih,iw,3),np.uint8)],[{'words':boxes}],True,True,True)[0]['words']
        boxes=[b.tolist() for b in boxes if b[2]>b[0] and b[3]>b[1]]
        cases.append(dict(prob=prob.flatten().tolist(),h=64,w=64,ih=ih,iw=iw,boxes=boxes))
    Path('testdata/rust_postprocess.json').write_text(json.dumps(cases,separators=(',',':')))

if __name__=='__main__':main()
