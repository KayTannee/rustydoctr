"""Controlled easy-page source-DPI, detector-size, and padding ablations."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR', str(ROOT/'.cache/doctr'))
import json
import time
import statistics
import hashlib
import numpy as np
import torch
import cv2
import pymupdf
from doctr.models import ocr_predictor
from pybaseline.metrics import score_words, detection_summary


def flatten(page):
    return [{'text': w['value'], 'polygon': w['geometry']} for b in page['blocks']
            for line in b['lines'] for w in line['words']]


def main():
    out = ROOT/'pybaseline/results/resolution'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); cv2.setNumThreads(1); torch.manual_seed(1729)
    manifest = json.loads((ROOT/'testdata/generated/manifest.json').read_text())
    page = manifest['pages'][0]
    assert page['id'] == 'a4_control'
    pdf_path = ROOT/'testdata/generated'/manifest['pdf']
    doc = pymupdf.open(pdf_path)
    images = {}
    for dpi in [100, 200, 400, 600]:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
        images[dpi] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    rows = []
    for det in ['fast_base', 'db_resnet34']:
        model = ocr_predictor(det, 'parseq', pretrained=True, assume_straight_pages=True,
                              preserve_aspect_ratio=True, det_bs=1, reco_bs=128,
                              detect_layout=False, detect_orientation=False).eval().cuda()
        detector = model.det_predictor
        shapes = []
        hook = detector.model.register_forward_pre_hook(lambda m, args: shapes.append(list(args[0].shape)))
        cases = [(600, size) for size in [1024, 1280, 1536, 2048, 2560, 3072]]
        cases += [(dpi, size) for dpi in [100, 200, 400] for size in [1024, 2048]]
        with torch.inference_mode():
            for dpi, size in cases:
                saved=out/f'{det}_dpi{dpi}_size{size}.json'
                if saved.exists():
                    rows.append(json.loads(saved.read_text())['result'])
                    continue
                detector.pre_processor.resize.size = (size, size)
                torch.cuda.reset_peak_memory_stats()
                prediction = model([images[dpi]]).export()['pages'][0]
                times = []
                for _ in range(3):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    model([images[dpi]])
                    torch.cuda.synchronize(); times.append(time.perf_counter()-start)
                words = flatten(prediction)
                scores = {str(t): detection_summary([score_words(page['words'], words, threshold=t)])
                          for t in [.25, .5, .75]}
                row = dict(det=det, source_dpi=dpi, source_shape=list(images[dpi].shape), size=size,
                           actual_tensor_shape=shapes[-1], scores=scores, trials_seconds=times,
                           median_seconds=statistics.median(times), allocated_peak_bytes=torch.cuda.max_memory_allocated())
                rows.append(row)
                (out/f'{det}_dpi{dpi}_size{size}.json').write_text(json.dumps(dict(result=row, words=words), indent=2))
                print(det, dpi, size, scores['0.5']['matched'], scores['0.5']['exact'], flush=True)
            hook.remove()
            # Paired raw-network ablation: exactly the same content pixels and scale.
            # Remove only right-hand padding, round width UP to a multiple of 32.
            # Bypass square-only predictor unpadding and map both outputs explicitly.
            for size in [1024, 2048]:
                detector.pre_processor.resize.size = (size, size)
                detector.pre_processor.resize.symmetric_pad = False
                assert not detector.pre_processor.resize.symmetric_pad
                tensor = detector.pre_processor([images[600]])[0].cuda()
                h, w = images[600].shape[:2]
                content_width = int(size*w/h)
                rect_width = ((content_width+31)//32)*32
                for width in [size, rect_width]:
                    x = tensor[:, :, :, :width].contiguous()
                    def infer():
                        return detector.model(x, return_preds=True)['preds'][0]
                    preds = infer(); times=[]
                    for _ in range(3):
                        torch.cuda.synchronize(); start=time.perf_counter(); infer()
                        torch.cuda.synchronize(); times.append(time.perf_counter()-start)
                    boxes = next(iter(preds.values())).copy()
                    boxes[:, [0,2]] *= width/size*h/w
                    words=[{'polygon':[[float(b[0]),float(b[1])],[float(b[2]),float(b[3])]]} for b in boxes]
                    row=dict(det=det, mode='padding_ablation', height=size, width=width,
                             actual_tensor_shape=list(x.shape), content_width=content_width,
                             score=detection_summary([score_words(page['words'], words)]),
                             median_seconds=statistics.median(times), trials_seconds=times)
                    rows.append(row)
                    print('padding',det,size,width,row['score']['matched'],flush=True)
        del model, detector, tensor, x
        torch.cuda.empty_cache()
    result=dict(page=page['id'], truth_words=len(page['words']), pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(), results=rows,
                method='Same vector PDF page; DPI changes rasterization only. Square sweep uses complete OCR with fixed PARSeq. Padding ablation uses raw detector+postprocessing only, identical content pixels and explicit coordinate remapping. One warmup and three synchronized timed repetitions per case. Synthetic single-page experiment, not population proof.')
    (out/'study.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
