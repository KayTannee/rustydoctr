"""Actual docTR OCRPredictor rotation ablation against saved native results."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
MODES={
    'doctr_upright':dict(assume_straight_pages=True,straighten_pages=False,detect_orientation=False),
    'doctr_crops':dict(assume_straight_pages=False,straighten_pages=False,detect_orientation=True),
    'doctr_straight_upright':dict(assume_straight_pages=True,straighten_pages=True,detect_orientation=True),
    'doctr_straight_crops':dict(assume_straight_pages=False,straighten_pages=True,detect_orientation=True),
    'doctr_default_upright':dict(assume_straight_pages=True,straighten_pages=True,detect_orientation=True),
    'doctr_default_crops':dict(assume_straight_pages=False,straighten_pages=True,detect_orientation=True),
    'doctr_default_no_crop':dict(assume_straight_pages=False,straighten_pages=True,detect_orientation=True,disable_crop_orientation=True),
}


def default_straightening_inverse(page,angle):
    """Observe the installed rotate_image/remove_image_padding coordinate operations."""
    import cv2
    import numpy as np
    from doctr.utils.geometry import compute_expanded_shape,rotate_image,remove_image_padding
    h,w=page.shape[:2];expand=h!=w
    hp=wp=0
    if expand:
        eh,ew=compute_expanded_shape((h,w),angle)
        hp=int(max(0,np.ceil(eh-h)));wp=int(max(0,np.ceil(ew-w)))
    ph,pw=h+hp,w+wp
    rot=np.vstack([cv2.getRotationMatrix2D((pw/2,ph/2),angle,1.),[0.,0.,1.]])
    pad1=np.array([[1.,0.,wp//2],[0.,1.,hp//2],[0.,0.,1.]])
    hp2=wp2=0
    if expand and h/w!=ph/pw:
        if ph/pw>h/w:wp2=int(ph*w/h-pw)
        else:hp2=int(pw*h/w-ph)
    pad2=np.array([[1.,0.,wp2//2],[0.,1.,hp2//2],[0.,0.,1.]])
    rotated=rotate_image(page,angle,expand=expand)
    rows=np.any(rotated,axis=(1,2));cols=np.any(rotated,axis=(0,2))
    cy=int(np.argmax(rows)) if rows.any() else 0;cx=int(np.argmax(cols)) if rows.any() else 0
    crop=np.array([[1.,0.,-cx],[0.,1.,-cy],[0.,0.,1.]])
    return remove_image_padding(rotated),np.linalg.inv(crop@pad2@rot@pad1)


def digest(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def evaluate(truth,words):
    from pybaseline.metrics import score_words
    scores={str(t):score_words(truth,words,t,True) for t in [.5,.25]}
    pairs=dict(scores['0.25']['pairs']);regions={}
    for region in sorted({w['region'] for w in truth}):
        indices=[i for i,w in enumerate(truth) if w['region']==region]
        regions[region]=dict(truth=len(indices),matched=sum(i in pairs for i in indices),exact=sum(i in pairs and words[pairs[i]]['text']==truth[i]['text'] for i in indices))
    singles=[dict(truth=w['text'],predicted=words[pairs[i]]['text'] if i in pairs else '<unmatched>',region=w['region']) for i,w in enumerate(truth) if len(w['text'])==1]
    for score in scores.values():del score['pairs']
    return dict(scores=scores,regions=regions,single_characters=singles)


def raster_audit(out,fixtures,manifest):
    import numpy as np
    from PIL import Image
    from doctr.utils.geometry import straighten_page
    pages={p['id']:p for p in manifest['pages']};audit=[]
    for row in json.loads((out/'doctr_default_upright.json').read_text()):
        if row['page']=='text_scale':continue
        with Image.open(fixtures/pages[row['page']]['image']) as source:image=np.array(source.convert('RGB'))
        angle=row['page_orientation']['value'];preserved,_=straighten_page(image,angle);default,_=default_straightening_inverse(image,angle)
        audit.append(dict(page=row['page'],angle=angle,preserve_shape=list(preserved.shape[:2]),default_shape=list(default.shape[:2]),
                          preserve_black_fraction=float(np.all(preserved==0,axis=2).mean()),default_black_fraction=float(np.all(default==0,axis=2).mean())))
    (out/'raster_audit.json').write_text(json.dumps(audit,indent=2))


def worker(mode,out):
    import cv2
    import doctr
    import numpy as np
    from PIL import Image
    import torch
    from doctr.models import ocr_predictor
    from doctr.models.preprocessor import PreProcessor
    from pybaseline.metrics import score_words
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    preserve=not mode.startswith('doctr_default')
    predictor=ocr_predictor(det_arch='db_resnet34',reco_arch='parseq',pretrained=True,
                            det_bs=1,reco_bs=256,preserve_aspect_ratio=True,symmetric_pad=True,
                            preserve_original_coords=preserve,export_as_straight_boxes=False,**MODES[mode]).eval().cuda()
    cfg=predictor.det_predictor.model.cfg
    predictor.det_predictor.pre_processor=PreProcessor((1536,1536),1,preserve_aspect_ratio=True,symmetric_pad=True,mean=cfg['mean'],std=cfg['std'])
    trace={};quads={}
    def detection_count(module,inputs):trace['detector_passes']=trace.get('detector_passes',0)+1
    predictor.det_predictor.model.register_forward_pre_hook(detection_count)
    original_angles=predictor._get_orientations
    def angles(pages,maps):
        coarse,combined=original_angles(pages,maps)
        trace.update(coarse=coarse,combined=combined)
        return coarse,combined
    predictor._get_orientations=angles
    original_straighten=predictor._straighten_pages
    def straighten(pages,maps,coarse=None,combined=None):
        fixed,inverses=original_straighten(pages,maps,coarse,combined)
        trace['straightened_shapes']=[list(p.shape[:2]) for p in fixed]
        if not preserve:
            observed=[]
            for page,result,angle in zip(pages,fixed,trace['combined']):
                reproduced,inverse=default_straightening_inverse(page,angle)
                assert np.array_equal(reproduced,result),'Default transform observation did not reproduce input pixels'
                observed.append(inverse.tolist())
            trace['default_inverse']=observed
        return fixed,inverses
    predictor._straighten_pages=straighten
    original_remap=predictor._remap_to_original_coords
    def remap(document,orig_shapes,straight_shapes,inverses,orig_pages=None):
        # Observation only: retain quadrilaterals before docTR reduces straight boxes
        # to axis-aligned envelopes. Uses its own inverse, never ground truth.
        for page,original,straight,inverse in zip(document.pages,orig_shapes,straight_shapes,inverses):
            oh,ow=original;sh,sw=straight
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        pts=np.asarray(word.geometry,dtype=np.float64).reshape(-1,2)
                        if len(pts)==2:
                            (x0,y0),(x1,y1)=pts;pts=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
                        xy=np.c_[pts*[sw,sh],np.ones(len(pts))]@inverse.T
                        xy[:,0]=xy[:,0].clip(0,ow-1)/ow;xy[:,1]=xy[:,1].clip(0,oh-1)/oh
                        quads[id(word)]=xy[:,:2].tolist()
        return original_remap(document,orig_shapes,straight_shapes,inverses,orig_pages)
    predictor._remap_to_original_coords=remap
    fixtures=ROOT/'output/pdf/quality';manifest=json.loads((fixtures/'manifest.json').read_text())
    pages=manifest['pages'];rows=[]
    with Image.open(fixtures/pages[0]['image']) as source:
        predictor([np.array(source.convert('RGB'))])
    torch.cuda.synchronize()
    for page in pages:
        trace.clear();quads.clear()
        image=np.array(Image.open(fixtures/page['image']).convert('RGB'))
        start=time.perf_counter();document=predictor([image]);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        result=document.pages[0];words=[]
        for block in result.blocks:
            for line in block.lines:
                for word in line.words:
                    returned=np.asarray(word.geometry).tolist()
                    mapped=quads.get(id(word),returned)
                    if not preserve:
                        pts=np.array(returned,dtype=np.float64)
                        if len(pts)==2:
                            (x0,y0),(x1,y1)=pts;pts=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
                        sh,sw=trace['straightened_shapes'][0];oh,ow=image.shape[:2]
                        xy=np.c_[pts*[sw,sh],np.ones(len(pts))]@np.array(trace['default_inverse'][0]).T
                        xy[:,0]=xy[:,0].clip(0,ow-1)/ow;xy[:,1]=xy[:,1].clip(0,oh-1)/oh
                        mapped=xy[:,:2].tolist()
                    words.append(dict(text=word.value,confidence=float(word.confidence),polygon=mapped,
                                      returned_polygon=returned,crop_orientation=word.crop_orientation))
        row=dict(mode=mode,page=page['id'],truth_angle=page['page_rotation_deg'],page_orientation=result.orientation,
                 trace=dict(trace),diagnostic_inference_seconds=elapsed,words=words,**evaluate(page['words'],words))
        row['returned_geometry_score']=score_words(page['words'],[dict(w,polygon=w['returned_polygon']) for w in words]) if preserve else None
        rows.append(row)
        print(mode,page['id'],'exact',row['scores']['0.5']['exact'],'angle',result.orientation,'det passes',trace['detector_passes'],flush=True)
        (out/f'{mode}.json').write_text(json.dumps(rows,indent=2))
    versions={'doctr':doctr.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'settings':MODES[mode],
              'detector_size':1536,'recognizer_batch':256,'device':'CUDA FP32, TF32 disabled; orientation classifiers on CUDA',
              'preserve_original_coords':preserve,'checkpoint_files':{p.name:digest(p) for p in (ROOT/'.cache/doctr').rglob('*.pt') if any(n in p.name for n in ['db_resnet34','parseq','orientation'])}}
    (out/f'{mode}_environment.json').write_text(json.dumps(versions,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=MODES)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--rust-results',type=Path,default=ROOT/'pybaseline/results/native_orientation_v1')
    args=parser.parse_args()
    out=(args.output or ROOT/'pybaseline/results'/('doctr_orientation_'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    if args.worker:
        worker(args.worker,out);return
    out.mkdir(parents=True,exist_ok=False)
    original=json.loads((args.rust_results/'provenance.json').read_text())
    for path in ['models/db_resnet34.onnx','models/parseq.onnx','models/page_orientation.onnx','output/pdf/quality/manifest.json']:
        expected=original.get(path,original.get(path.replace('/','\\')))
        assert expected==digest(ROOT/path),f'Stale Rust reference: {path}'
    fixtures=ROOT/'output/pdf/quality';manifest=json.loads((fixtures/'manifest.json').read_text())
    sources=[ROOT/'.venv-baseline/Lib/site-packages/doctr/models/_utils.py',ROOT/'.venv-baseline/Lib/site-packages/doctr/models/predictor/base.py',ROOT/'.venv-baseline/Lib/site-packages/doctr/models/predictor/pytorch.py',ROOT/'.venv-baseline/Lib/site-packages/doctr/utils/geometry.py']
    provenance={str(p.relative_to(ROOT)):digest(p) for p in sources}
    provenance.update({str((fixtures/p['image']).relative_to(ROOT)):digest(fixtures/p['image']) for p in manifest['pages']})
    (out/'provenance.json').write_text(json.dumps({'source_and_image_hashes':provenance,'rust_reference':str(args.rust_results.resolve()),'rust_provenance':original},indent=2))
    for mode in MODES:
        subprocess.run([sys.executable,'-u','-m','pybaseline.compare_doctr_orientation','--worker',mode,'--output',str(out)],cwd=ROOT,check=True,timeout=1200)
    rust=[]
    for mode in ['base','quarter','deskew','refined']:
        summary=json.loads((args.rust_results/mode/'summary.json').read_text())
        records={r['id']:r for r in summary['first_pages'].values()}
        for page in manifest['pages']:
            record=records[page['id']];words=[dict(w,polygon=w.get('quadrilateral',w['polygon'])) for w in record['words']]
            rust.append(dict(mode='rust_'+mode,page=page['id'],truth_angle=page['page_rotation_deg'],geometry=record.get('geometry'),words=words,**evaluate(page['words'],words)))
    (out/'rust.json').write_text(json.dumps(rust,indent=2))
    raster_audit(out,fixtures,manifest)
    subprocess.run([sys.executable,'-m','pybaseline.doctr_orientation_report',str(out)],cwd=ROOT,check=True)
    print('Completed accuracy comparison:',out,flush=True)


if __name__=='__main__':main()
