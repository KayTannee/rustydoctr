"""CPU-only page-orientation and fractional-skew diagnostics."""
import os,json
from pathlib import Path
os.environ.setdefault('DOCTR_CACHE_DIR',str(Path('.cache/doctr').resolve()))
import torch,numpy as np
from PIL import Image
from doctr.models import page_orientation_predictor
from pybaseline.quality_study import skew

def main():
 torch.set_num_threads(1);model=page_orientation_predictor(pretrained=True).eval().cpu()
 root=Path('output/pdf/quality');manifest=json.loads((root/'manifest.json').read_text());rows=[]
 for page in manifest['pages']:
  if page['family']=='text_scale':continue
  image=np.array(Image.open(root/page['image']).convert('RGB'))
  ids,angles,confidence=model([image]);residual,n=skew(image)
  row=dict(page=page['id'],truth_cw=page['page_rotation_deg'],classifier_angle=angles[0],confidence=confidence[0],raw_skew=residual,line_votes=n)
  rows.append(row);print(row,flush=True)
 Path('pybaseline/results/quality/orientation_cpu.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
