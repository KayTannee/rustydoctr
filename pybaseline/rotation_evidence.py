"""Save a concrete contextual I-to-dash example from the streaming warmup export."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np


def main():
    manifest=json.loads(Path('testdata/rotation/manifest.json').read_text())
    gt=manifest['pages'][0]
    sample_path=Path('pybaseline/results/streaming/tune_b1_r128/accuracy_sample.json')
    sample=json.loads(sample_path.read_text())
    target=next(w for w in gt['words'] if w['text']=='I' and w['sentence']=='It was cold outside I put my coat on')
    words=[w for b in sample['blocks'] for line in b['lines'] for w in line['words']]
    center=np.mean(target['polygon'],axis=0)
    candidate=min(words,key=lambda w:float(np.linalg.norm(np.mean(np.asarray(w['geometry']).reshape(-1,2),axis=0)-center)))
    image=Image.open(Path('testdata/rotation')/gt['image']).convert('RGB')
    draw=ImageDraw.Draw(image); scale=np.array(image.size)
    truth_pts=(np.array(target['polygon'])*scale).tolist()
    pred_pts=(np.array(candidate['geometry'])*scale).tolist()
    draw.line([tuple(p) for p in truth_pts+[truth_pts[0]]],fill='#008844',width=2)
    draw.line([tuple(p) for p in pred_pts+[pred_pts[0]]],fill='#cc3300',width=2)
    first_line=gt['words'][:9]
    points=np.concatenate([np.array(w['polygon'])*scale for w in first_line])
    bounds=(int(points[:,0].min())-15,int(points[:,1].min())-25,int(points[:,0].max())+15,int(points[:,1].max())+25)
    crop=image.crop(bounds)
    output=Path('pybaseline/results/rotation_screen_v2')
    crop.save(output/'contextual_I.png')
    evidence={'truth':target,'prediction':candidate,'selection':'Nearest predicted word centre to the first contextual I; manually inspect image for attribution.',
              'note':'Green: nominal ground-truth box; orange: predicted box. A local orientation of 0/180 alone cannot explain a vertical-to-horizontal change. This example must not be attributed solely to the rotation classifier.'}
    (output/'contextual_I.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({'truth':target['text'],'prediction':candidate['value'],'crop_orientation':candidate.get('crop_orientation'),'confidence':candidate['confidence']}))


if __name__=='__main__':
    main()
