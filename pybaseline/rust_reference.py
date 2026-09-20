"""Same-scope Python word pipeline: docTR/PyTorch or identical exported ONNX graphs."""
import argparse
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
import cv2
import numpy as np
import torch
from doctr.models.preprocessor import PreProcessor
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.detection._utils import _remove_padding
from doctr.models.recognition.parseq.pytorch import PARSeqPostProcessor
from doctr.utils.geometry import extract_crops


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['torch','ort'],required=True)
    p.add_argument('--image',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=1024)
    p.add_argument('--reco-batch',type=int,default=128)
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--dump',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    metadata=json.loads((ROOT/'models/metadata.json').read_text())
    det_pre=PreProcessor((a.size,a.size),1,preserve_aspect_ratio=True,symmetric_pad=True,
                         mean=metadata['db_resnet34']['mean'],std=metadata['db_resnet34']['std'])
    reco_pre=PreProcessor((32,128),a.reco_batch,preserve_aspect_ratio=True,symmetric_pad=False,
                          mean=metadata['parseq']['mean'],std=metadata['parseq']['std'])
    det_post=DBPostProcessor(assume_straight_pages=True)
    reco_post=PARSeqPostProcessor(vocab=metadata['parseq']['vocab'])
    start=time.perf_counter()
    if a.backend=='ort':
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        ort.preload_dlls(directory=str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib'))
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
        providers=[('CUDAExecutionProvider',{'use_tf32':'0'})]
        det=ort.InferenceSession(str(ROOT/'models/db_resnet34.onnx'),sess_options=options,providers=providers)
        reco=ort.InferenceSession(str(ROOT/'models/parseq.onnx'),sess_options=options,providers=providers)
        assert det.get_providers()[0]==reco.get_providers()[0]=='CUDAExecutionProvider'
        def detect(x):return 1/(1+np.exp(-det.run(None,{'input':x.numpy()})[0]))
        def recognize(x):return torch.from_numpy(reco.run(None,{'input':x.numpy()})[0])
    else:
        from doctr.models import db_resnet34,parseq
        det=db_resnet34(pretrained=True,exportable=True).eval().cuda()
        reco=parseq(pretrained=True).eval().cuda()
        def detect(x):return torch.sigmoid(det(x.cuda())['logits']).cpu().numpy()
        def recognize(x):
            features=reco.feat_extractor(x.cuda())['features'][:,1:,:]
            return reco.decode_autoregressive(features).cpu()
    load_seconds=time.perf_counter()-start
    images=[cv2.cvtColor(cv2.imread(str(x)),cv2.COLOR_BGR2RGB) for x in a.image]
    def cycle(dump=False):
        pages=[]
        for index,(path,img) in enumerate(zip(a.image,images)):
            x=det_pre([img])[0]
            prob=detect(x)
            boxes=det_post(prob.transpose(0,2,3,1))[0][0]
            boxes=_remove_padding([img],[{'words':boxes}],True,True,True)[0]['words']
            crops=extract_crops(img,boxes[:,:4])
            assert all(c.shape[1]/c.shape[0]<=8 for c in crops),'Wide crop outside current Rust scope'
            texts=[];batches=reco_pre(crops) if crops else []
            for batch in batches:texts.extend(reco_post(recognize(batch)))
            if dump and index==0:
                x.numpy().tofile(a.output/'detector_input.f32');prob.tofile(a.output/'detector_probability.f32')
                if batches:batches[0].numpy().tofile(a.output/'recognizer_input.f32')
            words=[dict(polygon=[[float(b[0]),float(b[1])],[float(b[2]),float(b[3])]],objectness=float(b[4]),text=t,confidence=float(c)) for b,(t,c) in zip(boxes,texts)]
            pages.append(dict(image=str(path),words=words))
        return pages
    with torch.inference_mode():
        start=time.perf_counter();pages=cycle(a.dump);warmup=time.perf_counter()-start;trials=[]
        while len(trials)<a.repeats or sum(trials)<a.seconds:
            start=time.perf_counter();cycle();trials.append(time.perf_counter()-start)
            print(a.backend,len(trials),trials[-1],flush=True)
    result=dict(backend=a.backend,size=a.size,reco_batch=a.reco_batch,model_load_seconds=load_seconds,
                warmup_seconds=warmup,trials_seconds=trials,pages_per_second=len(images)*len(trials)/sum(trials),pages=pages,
                scope='Same upright word-only pipeline as Rust; image decode/model load/warmup/JSON writing excluded. ORT uses full-length exported PARSeq; PyTorch uses native early exit.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
