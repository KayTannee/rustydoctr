"""Separate compatibility diagnostic: copy crop arrays without changing pixels/OCR rules."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
from scripts.tune_dense_detection import read,write

def main():
    import cv2,numpy as np,torch
    from PIL import Image
    from doctr.models import ocr_predictor
    from doctr.models.preprocessor import PreProcessor
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=ocr_predictor(det_arch='db_resnet34',reco_arch='parseq',pretrained=True,det_bs=1,reco_bs=256,preserve_aspect_ratio=True,symmetric_pad=True,assume_straight_pages=False,straighten_pages=True,detect_orientation=True,preserve_original_coords=True,export_as_straight_boxes=False).eval().cuda()
    cfg=model.det_predictor.model.cfg;model.det_predictor.pre_processor=PreProcessor((1536,1536),1,preserve_aspect_ratio=True,symmetric_pad=True,mean=cfg['mean'],std=cfg['std'])
    original=model.reco_predictor.pre_processor.sample_transforms
    def copy_crop(x):return original(x.copy(order='C') if isinstance(x,np.ndarray) else x)
    model.reco_predictor.pre_processor.sample_transforms=copy_crop
    with torch.inference_mode():
        for dataset in ['cord-v2','hiertext']:
            source=ROOT/'pybaseline/results/public_v1'/dataset/'python_rotated_1536';out=source.parent/'python_rotated_copy_1536';out.mkdir(exist_ok=True)
            if (out/'summary.json').exists():continue
            pages={p['id']:p for p in read(ROOT/'testdata/public'/dataset/'prepared/manifest.json')};rows=[json.loads(l) for l in (source/'pages.jsonl').read_text(encoding='utf-8').splitlines()];controls=[r['id'] for r in rows if not r.get('error')][:3];repaired=[];start=time.perf_counter()
            for row in rows:
                if not row.get('error') and row['id'] not in controls:continue
                with Image.open(pages[row['id']]['image']) as im:image=np.array(im.convert('RGB'))
                result=model([image]).pages[0];words=[dict(text=w.value,confidence=float(w.confidence),polygon=np.asarray(w.geometry).tolist()) for b in result.blocks for line in b.lines for w in line.words]
                if row['id'] in controls:
                    assert [w['text'] for w in words]==[w['text'] for w in row['words']],row['id']
                    np.testing.assert_allclose([w['polygon'] for w in words],[w['polygon'] for w in row['words']],atol=1e-6,rtol=0)
                else:
                    repaired.append(dict(page=row['id'],previous_error=row['error'],words=len(words)));row.update(words=words,orientation=result.orientation,error=None,compatibility_copy=True)
                    if len(repaired)%20==0:print(dataset,'repaired',len(repaired),flush=True)
            with (out/'pages.jsonl').open('w',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            write(out/'repair.json',dict(repaired=repaired,unchanged_controls=controls,diagnostic_seconds=time.perf_counter()-start,policy='Only recognizer input array layout changed via copy(order=C). Successful original pages reused; no complete throughput measurement.'))
            write(out/'summary.json',dict(pages=len(rows),failed_pages=[],wall_seconds=None,pages_per_second=None,resources={},diagnostic_only=True,source_summary=str(source/'summary.json'),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            print(dataset,'repaired',len(repaired),'controls passed',len(controls),flush=True)
if __name__=='__main__':main()
