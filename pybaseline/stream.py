"""Bounded decode -> full docTR OCR -> JSON writer streaming benchmark.

Exactly one inference process. Feeder and postprocessor are separate CPU processes.
Only descriptors traverse the input queue; page pixels use a bounded shared-memory pool.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('testdata/rotation/manifest.json'))
    p.add_argument('--page',default='upright_characters')
    p.add_argument('--selection',type=Path)
    p.add_argument('--det',default='db_resnet34')
    p.add_argument('--reco',default='crnn_mobilenet_v3_large')
    p.add_argument('--rotation',choices=['straight','crops','straighten','straighten_no_crop'],default='crops')
    p.add_argument('--size',type=int,default=1536)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--reco-batch',type=int,default=256)
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.selection:
        selected=json.loads(args.selection.read_text())
        for key in ['det','reco','rotation','size']:
            setattr(args,key,selected[key])
    if min(args.batch,args.reco_batch,args.seconds,args.threads)<=0 or args.size%32:
        p.error('positive sizes/duration/threads required; detector size must be divisible by 32')
    if args.output.exists():
        p.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    import cv2
    import numpy as np
    import torch
    import doctr
    from doctr.models import ocr_predictor
    from pybaseline.monitor import Monitor
    from pybaseline.stream_workers import feed,consume
    from pybaseline.metrics import score_words
    torch.set_num_threads(args.threads); cv2.setNumThreads(args.threads)
    torch.manual_seed(1729)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    manifest=json.loads(args.manifest.read_text())
    page=next((page for page in manifest['pages'] if page['id']==args.page),None)
    if page is None:
        raise ValueError(f'Unknown page: {args.page}')
    image_path=args.manifest.parent/page['image']
    image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    model=ocr_predictor(args.det,args.reco,pretrained=True,assume_straight_pages=args.rotation=='straight',
                        straighten_pages=args.rotation in ['straighten','straighten_no_crop'],
                        disable_crop_orientation=args.rotation=='straighten_no_crop',
                        preserve_original_coords=True,det_bs=args.batch,reco_bs=args.reco_batch,
                        detect_layout=False,detect_tables=False).eval().cuda()
    model.det_predictor.pre_processor.resize.size=(args.size,args.size)
    warm_start=time.perf_counter()
    with torch.inference_mode():
        for _ in range(2):
            warm=model([image]*args.batch)
        torch.cuda.synchronize()
    sample=warm.export()['pages'][0]
    (args.output/'accuracy_sample.json').write_text(json.dumps(sample),encoding='utf-8')
    sample_words=[{'text':w['value'],'polygon':w['geometry']} for b in sample['blocks'] for line in b['lines'] for w in line['words']]
    sample_accuracy=score_words(page['words'],sample_words)
    del warm
    warmup=time.perf_counter()-warm_start
    slots=args.batch*2
    ctx=mp.get_context('spawn')
    free=ctx.Queue(maxsize=slots); ready=ctx.Queue(maxsize=slots); results=ctx.Queue(maxsize=2)
    start=ctx.Event(); stop=ctx.Event()
    shm=shared_memory.SharedMemory(create=True,size=slots*image.nbytes)
    buffers=np.ndarray((slots,*image.shape),dtype=np.uint8,buffer=shm.buf)
    for i in range(slots):
        free.put(i)
    feeder=ctx.Process(target=feed,args=(str(image_path),shm.name,image.shape,slots,free,ready,start,stop,args.seconds,str(args.output/'feeder.json')),name='page-feeder')
    post=ctx.Process(target=consume,args=(results,start,str(args.output)),name='result-writer')
    inference_seconds=export_seconds=input_wait_seconds=output_wait_seconds=0.0
    batches=[]; processed=0; ended=False
    monitor=Monitor(args.output/'telemetry.jsonl',.05)
    cpu_start=time.process_time()
    try:
        feeder.start(); post.start()
        with monitor,torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            start_ns=time.time_ns(); start.set()
            while not ended:
                metadata=[]
                waited=time.perf_counter()
                while len(metadata)<args.batch:
                    try:
                        meta=ready.get(timeout=1)
                    except queue.Empty:
                        if feeder.exitcode is not None:
                            raise RuntimeError(f'Feeder exited without end marker: {feeder.exitcode}')
                        if post.exitcode is not None:
                            raise RuntimeError(f'Postprocessor exited early: {post.exitcode}')
                        continue
                    if meta is None:
                        ended=True; break
                    metadata.append(meta)
                input_wait_seconds+=time.perf_counter()-waited
                if not metadata:
                    break
                images=[buffers[meta['slot']] for meta in metadata]
                torch.cuda.synchronize(); began=time.perf_counter()
                prediction=model(images)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-began; inference_seconds+=elapsed
                completed_ns=time.time_ns()
                began=time.perf_counter(); exported=prediction.export()['pages']; export_seconds+=time.perf_counter()-began
                for meta in metadata:
                    meta['inference_complete_ns']=completed_ns
                    free.put(meta['slot'])
                waited=time.perf_counter()
                while True:
                    try:
                        results.put((metadata,exported),timeout=1); break
                    except queue.Full:
                        if post.exitcode is not None:
                            raise RuntimeError(f'Postprocessor failed: {post.exitcode}')
                output_wait_seconds+=time.perf_counter()-waited
                processed+=len(metadata)
                batches.append({'pages':len(metadata),'inference_seconds':elapsed,'completed_ns':completed_ns})
                if len(batches)%10==0:
                    print(f'{processed} pages; {(time.time_ns()-start_ns)/1e9:.1f}s elapsed',flush=True)
                del prediction,images,exported
            # Keep monitoring until queued outputs are durably written.
            while True:
                try:
                    results.put(None,timeout=1); break
                except queue.Full:
                    if post.exitcode is not None:
                        raise RuntimeError('Postprocessor failed during drain')
            feeder.join(30); post.join(60)
            if feeder.exitcode!=0 or post.exitcode!=0:
                raise RuntimeError(f'Pipeline worker failed: feeder={feeder.exitcode}, post={post.exitcode}')
            end_ns=time.time_ns()
        feed_stats=json.loads((args.output/'feeder.json').read_text())
        post_stats=json.loads((args.output/'post.json').read_text())
        assert processed==feed_stats['pages']==post_stats['pages'], 'Lost pages'
        summary={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 'environment':{'doctr':doctr.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name()},
                 'corpus_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                 'page_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                 'pages':processed,'page_words':len(page['words']),'unique_pages':1,'warmup_seconds':warmup,
                 'accuracy_sample':sample_accuracy,
                 'start_ns':start_ns,'end_ns':end_ns,'pipeline_wall_seconds':(end_ns-start_ns)/1e9,
                 'pages_per_second':processed/((end_ns-start_ns)/1e9),
                 'post_makespan_pages_per_second':processed/post_stats['end_to_end_wall_seconds'],
                 'engine_cpu_seconds':time.process_time()-cpu_start,'inference_seconds':inference_seconds,
                 'export_seconds':export_seconds,'input_wait_seconds':input_wait_seconds,'output_wait_seconds':output_wait_seconds,
                 'shared_memory_bytes':slots*image.nbytes,'queue_slots':slots,'output_queue_batches':2,
                 'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
                 'resources':monitor.summary(),'feeder':feed_stats,'post':post_stats,'batches':batches,
                 'scope':'Warm-cache repeated PNG decode, shared-memory feeder, complete docTR OCR and document construction, export, IPC, JSONL serialization/write, final fsync and drain. Model loading/warmup excluded. One inference process; backpressure, no dropped pages.'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({'pages':processed,'pages_per_second':summary['pages_per_second'],'output':str(args.output)}),flush=True)
    finally:
        stop.set(); start.set()
        for worker in [feeder,post]:
            if worker.pid and worker.is_alive():
                worker.terminate(); worker.join(10)
        shm.close(); shm.unlink()
        for channel in [free,ready,results]:
            channel.cancel_join_thread(); channel.close()


if __name__=='__main__':
    main()
