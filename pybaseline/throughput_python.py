"""Bounded input-prefetch Python reference for the throughput experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['torch','ort'],required=True)
    p.add_argument('--workload',type=Path,default=Path('testdata/throughput.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=1536)
    p.add_argument('--page-batch',type=int,default=2)
    p.add_argument('--reco-batch',type=int,default=256)
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--pages',type=int,default=0)
    p.add_argument('--arena-mib',type=int,default=8192)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from doctr.models.preprocessor import PreProcessor
    from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
    from doctr.models.detection._utils import _remove_padding
    from doctr.models.recognition.parseq.pytorch import PARSeqPostProcessor
    from doctr.models.recognition.predictor._utils import split_crops,remap_preds
    from doctr.utils.geometry import extract_crops
    from pybaseline.monitor import Monitor
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=json.loads((ROOT/'models/metadata.json').read_text())
    workload=json.loads(a.workload.read_text());paths=[Path(x['image']) for x in workload['pages']]
    det_pre=PreProcessor((a.size,a.size),a.page_batch,preserve_aspect_ratio=True,symmetric_pad=True,mean=meta['db_resnet34']['mean'],std=meta['db_resnet34']['std'])
    reco_pre=PreProcessor((32,128),a.reco_batch,preserve_aspect_ratio=True,symmetric_pad=False,mean=meta['parseq']['mean'],std=meta['parseq']['std'])
    det_post=DBPostProcessor(assume_straight_pages=True);reco_post=PARSeqPostProcessor(vocab=meta['parseq']['vocab'])
    loading=time.perf_counter()
    if a.backend=='ort':
        import onnxruntime as ort
        ort.set_default_logger_severity(3);ort.preload_dlls(directory=str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib'))
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
        def session(name,fraction):
            return ort.InferenceSession(str(ROOT/'models'/f'{name}.onnx'),sess_options=options,providers=[('CUDAExecutionProvider',{
                'use_tf32':'0','cudnn_conv_algo_search':'HEURISTIC','gpu_mem_limit':str(int(a.arena_mib*2**20*fraction))})])
        det=session('db_resnet34',.25);reco=session('parseq',.75)
        assert det.get_providers()[0]==reco.get_providers()[0]=='CUDAExecutionProvider'
        def detect(x):return 1/(1+np.exp(-det.run(None,{'input':x.numpy()})[0]))
        def recognize(x):return torch.from_numpy(reco.run(None,{'input':x.numpy()})[0])
    else:
        from doctr.models import db_resnet34,parseq
        det=db_resnet34(pretrained=True,exportable=True).eval().cuda();reco=parseq(pretrained=True).eval().cuda()
        def detect(x):return torch.sigmoid(det(x.cuda())['logits']).cpu().numpy()
        def recognize(x):return reco.decode_autoregressive(reco.feat_extractor(x.cuda())['features'][:,1:,:]).cpu()
    load_seconds=time.perf_counter()-loading
    def decode(indices):
        admitted=time.perf_counter()
        images=[cv2.cvtColor(cv2.imread(str(paths[i%len(paths)])),cv2.COLOR_BGR2RGB) for i in indices]
        return indices,images,admitted
    stages={k:0. for k in ['detector_preprocess','detection','geometry_crops','recognition_preprocess','recognition','decode_text']}
    def process(images):
        t=time.perf_counter();prepared=det_pre(images)[0];stages['detector_preprocess']+=time.perf_counter()-t
        t=time.perf_counter();pred=detect(prepared);stages['detection']+=time.perf_counter()-t
        t=time.perf_counter()
        raw=det_post(pred.transpose(0,2,3,1))
        boxes=[d['words'] for d in _remove_padding(images,[{'words':b[0]} for b in raw],True,True,True)]
        crops=[c for img,b in zip(images,boxes) for c in extract_crops(img,b[:,:4])]
        crops,mapping,remapped=split_crops(crops,8,6,.5)
        stages['geometry_crops']+=time.perf_counter()-t
        texts=[]
        t=time.perf_counter();batches=reco_pre(crops) if crops else [];stages['recognition_preprocess']+=time.perf_counter()-t
        for batch in batches:
            t=time.perf_counter();logits=recognize(batch);stages['recognition']+=time.perf_counter()-t
            t=time.perf_counter();texts.extend(reco_post(logits));stages['decode_text']+=time.perf_counter()-t
        if remapped:texts=remap_preds(texts,mapping,.5)
        pages=[];offset=0
        for b in boxes:
            page=[dict(polygon=[[float(x[0]),float(x[1])],[float(x[2]),float(x[3])]],objectness=float(x[4]),text=t,confidence=float(c)) for x,(t,c) in zip(b,texts[offset:offset+len(b)])]
            pages.append(page);offset+=len(b)
        return pages
    with torch.inference_mode():
        warm=time.perf_counter()
        for i in range(0,len(paths),a.page_batch):process(decode(list(range(i,min(i+a.page_batch,len(paths)))))[1])
        warmup_seconds=time.perf_counter()-warm
        stages={k:0. for k in stages}
        print('Warmup complete; starting measurement',flush=True)
        latencies=[];count=0;unique={};timeline=[]
        with Monitor(a.output/'telemetry.jsonl') as monitor:
            start_ns=time.time_ns();start=time.perf_counter();next_id=0;done=False
            with ThreadPoolExecutor(max_workers=1) as pool,(a.output/'pages.jsonl').open('w',encoding='utf-8') as output:
                pending=deque()
                while not done or pending:
                    while not done and len(pending)<2:
                        if a.pages and next_id>=a.pages:done=True;break
                        if not a.pages and next_id%len(paths)==0 and next_id>0 and time.perf_counter()-start>=a.seconds:done=True;break
                        n=min(a.page_batch,a.pages-next_id) if a.pages else min(a.page_batch,len(paths)-next_id%len(paths))
                        pending.append(pool.submit(decode,list(range(next_id,next_id+n))));next_id+=n
                    if not pending:break
                    indices,images,admitted=pending.popleft().result();pages=process(images)
                    for i,words in zip(indices,pages):
                        assert i==count
                        record={'sequence':i,'page':i%len(paths),'words':words}
                        output.write(json.dumps(record,separators=(',',':'))+'\n')
                        unique.setdefault(str(i%len(paths)),record);count+=1
                    output.flush();latencies.extend([time.perf_counter()-admitted]*len(pages))
                    timeline.append({'elapsed':time.perf_counter()-start,'pages':count})
                    if count%len(paths)==0:print(count,'pages',round(time.perf_counter()-start,1),'s',flush=True)
                output.flush();os.fsync(output.fileno())
            elapsed=time.perf_counter()-start;end_ns=time.time_ns()
        result={'implementation':'python_'+a.backend,'config':vars(a),'model':'db_resnet34 + parseq','pages':count,'wall_seconds':elapsed,'pages_per_second':count/elapsed,
                'model_load_seconds':load_seconds,'warmup_seconds':warmup_seconds,'start_ns':start_ns,'end_ns':end_ns,
                'corpus_sha256':hashlib.sha256(a.workload.read_bytes()).hexdigest(),'resources':monitor.summary(),
                'latency_seconds':{f'p{p}':float(np.percentile(latencies,p)) for p in [50,95,99]},'timeline':timeline,'first_pages':unique,'stage_seconds':stages,
                'scope':'Warm-cache decode with 2-batch bounded prefetch, upright full word OCR including wide-crop splitting, JSONL flush/fsync. Load/warmup excluded; no orientation or document layout.'}
        (a.output/'summary.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
        print('DONE',count/elapsed,'pages/s',flush=True)

if __name__=='__main__':main()
