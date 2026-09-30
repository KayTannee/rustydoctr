"""Stock PyTorch docTR public-dataset worker (no bespoke OCR corrections)."""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=1024);p.add_argument('--rotate',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import cv2, doctr, numpy as np, torch
    from PIL import Image
    from doctr.models import ocr_predictor
    from doctr.models.preprocessor import PreProcessor
    from pybaseline.monitor import Monitor
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();pages=json.loads(a.manifest.read_text(encoding='utf-8'))
    with Monitor(a.output/'telemetry.jsonl',interval=.1):
        loading=time.perf_counter()
        model=ocr_predictor(det_arch='db_resnet34',reco_arch='parseq',pretrained=True,det_bs=1,reco_bs=256,preserve_aspect_ratio=True,symmetric_pad=True,assume_straight_pages=not a.rotate,straighten_pages=a.rotate,detect_orientation=a.rotate,preserve_original_coords=True,export_as_straight_boxes=False).eval().cuda()
        cfg=model.det_predictor.model.cfg
        model.det_predictor.pre_processor=PreProcessor((a.size,a.size),1,preserve_aspect_ratio=True,symmetric_pad=True,mean=cfg['mean'],std=cfg['std'])
        load=time.perf_counter()-loading
        def decode(page):
            with Image.open(page['image']) as im:return np.array(im.convert('RGB'))
        def process(image):
            try:result=model([image]).pages[0]
            except ValueError as exc:
                if 'stride' not in str(exc) or 'negative' not in str(exc):raise
                return [],None,str(exc)
            words=[dict(text=w.value,confidence=float(w.confidence),polygon=np.asarray(w.geometry).tolist()) for b in result.blocks for line in b.lines for w in line.words]
            return words,result.orientation,None
        with torch.inference_mode():
            warm=time.perf_counter()
            # Same full-corpus warmup as the native throughput runner.
            for i,page in enumerate(pages):
                process(decode(page))
                if (i+1)%100==0:print('Warmup',i+1,'/',len(pages),flush=True)
            torch.cuda.synchronize();warm=time.perf_counter()-warm
            print('Warmup complete',round(warm,1),'s',flush=True)
            start_ns=time.time_ns();start=time.perf_counter();count=0;failures=[]
            with ThreadPoolExecutor(max_workers=1) as pool,(a.output/'pages.jsonl').open('w',encoding='utf-8') as f:
                pending=deque();next_page=0
                while next_page<len(pages) or pending:
                    while next_page<len(pages) and len(pending)<3:
                        pending.append((next_page,pool.submit(decode,pages[next_page])));next_page+=1
                    index,future=pending.popleft();words,orientation,error=process(future.result());torch.cuda.synchronize()
                    if error:failures.append(dict(page=pages[index]['id'],error=error));print('Page inference failure',pages[index]['id'],error,flush=True)
                    f.write(json.dumps(dict(id=pages[index]['id'],sequence=index,words=words,orientation=orientation,error=error),ensure_ascii=False)+'\n');count+=1
                    if count%25==0:print(count,'/',len(pages),'pages;',round(time.perf_counter()-start,1),'s',flush=True)
                f.flush();os.fsync(f.fileno())
            elapsed=time.perf_counter()-start;end_ns=time.time_ns()
        checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'.cache/doctr').rglob('*.pt') if any(n in p.name for n in ['db_resnet34','parseq','orientation'])}
        result=dict(failed_pages=failures,worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pages=count,wall_seconds=elapsed,pages_per_second=count/elapsed,start_ns=start_ns,end_ns=end_ns,model_load_seconds=load,warmup_seconds=warm,config=dict(size=a.size,rotate=a.rotate,reco_batch=256,det_batch=1,prefetch=3),versions=dict(doctr=doctr.__version__,torch=torch.__version__),checkpoints=checkpoints)
    telemetry=[json.loads(line) for line in (a.output/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()];measured=[r for r in telemetry if start_ns<=r['time_ns']<=end_ns];idle=next((r['device_vram_bytes'] for r in telemetry if r.get('device_vram_bytes') is not None),None)
    result['resources']={k+'_max':max((r[k] for r in measured if r.get(k) is not None),default=None) for k in ['device_vram_bytes','rss_bytes']}
    result['resources']['vram_increment_peak_bytes']=max(r.get('device_vram_bytes',0) or 0 for r in telemetry)-idle if idle is not None else None
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('DONE',count,'pages',round(count/elapsed,3),'pages/s',flush=True)
if __name__=='__main__':main()
