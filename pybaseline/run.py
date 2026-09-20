"""Run one model in a fresh process. See README for matrix and metric definitions."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR', str(ROOT / '.cache/doctr'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('USE_TORCH', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['detection', 'recognition', 'ocr'], required=True)
    parser.add_argument('--det', default='fast_base')
    parser.add_argument('--reco', default='parseq')
    parser.add_argument('--manifest', type=Path, default=ROOT/'testdata/generated/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2, help='page/detector batch size')
    parser.add_argument('--reco-batch', type=int, default=128)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--rotation', choices=['straight', 'crops', 'straighten', 'straighten_no_crop'], default='straight')
    parser.add_argument('--stretch', action='store_true')
    parser.add_argument('--min-seconds', type=float, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--sample-ms', type=float, default=50)
    parser.add_argument('--profile', action='store_true', help='extra untimed cProfile and torch profiler pass')
    parser.add_argument('--gate', type=Path, help='optional shared start file for concurrency experiments')
    args = parser.parse_args()
    if min(args.batch, args.reco_batch, args.repeats, args.threads, args.size, args.sample_ms) <= 0:
        parser.error('batch sizes, repeats, threads, size and sample-ms must be positive')
    if args.min_seconds < 0 or args.size % 32:
        parser.error('min-seconds must be nonnegative and detector size must be divisible by 32')
    args.output.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    import torch
    import doctr
    from doctr.models import detection_predictor, recognition_predictor, ocr_predictor
    from pybaseline.metrics import score_words, detection_summary, recognition_score
    from pybaseline.monitor import Monitor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required: refusing to label CPU results as a GPU baseline')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.manual_seed(1729)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    started = time.perf_counter()
    images = [cv2.cvtColor(cv2.imread(str(args.manifest.parent/p['image'])), cv2.COLOR_BGR2RGB) for p in manifest['pages']]
    load_seconds = time.perf_counter()-started
    crop_labels, crop_pages = [], []
    inputs = images
    if args.kind == 'recognition':
        inputs = []
        for page, image in zip(manifest['pages'], images):
            h, w = image.shape[:2]
            for word in page['words']:
                points = np.array(word['polygon'], dtype=np.float32)*[w, h]
                cw = max(2, round(float(np.linalg.norm(points[1]-points[0]))))
                ch = max(2, round(float(np.linalg.norm(points[3]-points[0]))))
                transform = cv2.getPerspectiveTransform(points.astype(np.float32), np.float32([[2,2],[cw+2,2],[cw+2,ch+2],[2,ch+2]]))
                inputs.append(cv2.warpPerspective(image, transform, (cw+4,ch+4), borderMode=cv2.BORDER_REPLICATE))
                crop_labels.append(word['text']); crop_pages.append(page['id'])
    preparation_seconds = time.perf_counter()-started
    started = time.perf_counter()
    straight = args.rotation == 'straight'
    if args.kind == 'detection':
        model = detection_predictor(args.det, pretrained=True, assume_straight_pages=straight,
                                    batch_size=args.batch, preserve_aspect_ratio=not args.stretch)
    elif args.kind == 'recognition':
        model = recognition_predictor(args.reco, pretrained=True, batch_size=args.reco_batch)
    else:
        model = ocr_predictor(args.det, args.reco, pretrained=True, assume_straight_pages=straight,
                              straighten_pages=args.rotation in ['straighten','straighten_no_crop'], preserve_original_coords=True,
                              disable_crop_orientation=args.rotation == 'straighten_no_crop',
                              preserve_aspect_ratio=not args.stretch, det_bs=args.batch, reco_bs=args.reco_batch,
                              detect_layout=False, detect_orientation=False)
    if args.kind != 'recognition':
        detector = model if args.kind == 'detection' else model.det_predictor
        detector.pre_processor.resize.size = (args.size, args.size)
    model = model.eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter()-started
    chunk_size = max(args.reco_batch, 512) if args.kind == 'recognition' else args.batch

    def cycle(collect=False):
        predictions = []
        for offset in range(0, len(inputs), chunk_size):
            result = model(inputs[offset:offset+chunk_size])
            if collect:
                if args.kind == 'ocr':
                    predictions.extend(result.export()['pages'])
                else:
                    predictions.extend(result)
        return predictions

    # Full-corpus warmup handles heterogeneous sizes and recognition sequence lengths.
    with torch.inference_mode():
        started = time.perf_counter()
        predictions = cycle(True)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter()-started
        if args.gate:
            (args.output/'ready').touch()
            deadline = time.monotonic()+600
            while not args.gate.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Concurrency start gate timed out')
                time.sleep(0.05)
        torch.cuda.reset_peak_memory_stats()
        trials = []
        monitor = Monitor(args.output/'telemetry.jsonl', args.sample_ms/1000)
        with monitor:
            measured_start_ns = time.time_ns()
            while len(trials) < args.repeats or sum(trials) < args.min_seconds:
                torch.cuda.synchronize()
                start = time.perf_counter()
                cycle()
                torch.cuda.synchronize()
                trials.append(time.perf_counter()-start)
                print(f'{args.kind} {args.det if args.kind == "detection" else args.reco}: trial {len(trials)} {trials[-1]:.3f}s', flush=True)
            measured_end_ns = time.time_ns()
        memory = {'allocated_peak_bytes': torch.cuda.max_memory_allocated(), 'reserved_peak_bytes': torch.cuda.max_memory_reserved()}
    by_page = []
    serialized = []
    if args.kind == 'recognition':
        texts = [p[0] for p in predictions]
        accuracy = recognition_score(crop_labels, texts)
        for page in manifest['pages']:
            indices = [i for i, name in enumerate(crop_pages) if name == page['id']]
            by_page.append(dict(page=page['id'], **recognition_score([crop_labels[i] for i in indices], [texts[i] for i in indices])))
        serialized = [{'truth': a, 'text': b[0], 'confidence': float(b[1]), 'page': p} for a,b,p in zip(crop_labels,predictions,crop_pages)]
    else:
        for page, prediction in zip(manifest['pages'], predictions):
            words = []
            if args.kind == 'ocr':
                for block in prediction['blocks']:
                    for line in block['lines']:
                        for word in line['words']:
                            words.append({'polygon': word['geometry'], 'text': word['value'], 'confidence': word['confidence']})
            else:
                for box in next(iter(prediction.values())):
                    if box.ndim == 1:
                        x0,y0,x1,y1,confidence = box
                        poly = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                    else:
                        poly, confidence = box[:4], box[4,1]
                    words.append({'polygon': np.asarray(poly).tolist(), 'confidence': float(confidence)})
            by_page.append(dict(page=page['id'], **score_words(page['words'], words)))
            serialized.append({'page': page['id'], 'words': words})
        accuracy = detection_summary(by_page)
    result = {'schema_version': 1, 'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'doctr': doctr.__version__,
                              'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                              'cpu': platform.processor(), 'logical_cpus': os.cpu_count()},
              'corpus_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'pages': len(images), 'items_per_trial': len(inputs), 'trials_seconds': trials,
              'throughput_per_second': len(inputs)*len(trials)/sum(trials),
              'trial_seconds_median': float(np.median(trials)), 'trial_seconds_p95': float(np.percentile(trials,95)),
              'load_images_seconds': load_seconds, 'prepare_inputs_seconds': preparation_seconds,
              'model_load_seconds_including_download': model_load_seconds, 'warmup_seconds': warmup_seconds,
              'measured_start_ns': measured_start_ns, 'measured_end_ns': measured_end_ns,
              'memory': memory, 'resources': monitor.summary(), 'accuracy': accuracy, 'by_page': by_page}
    (args.output/'predictions.json').write_text(json.dumps(serialized), encoding='utf-8')
    (args.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.profile:
        # Separate diagnostic pass: profiler overhead never contaminates throughput trials.
        prof = cProfile.Profile()
        prof.enable()
        with torch.inference_mode():
            cycle()
            torch.cuda.synchronize()
        prof.disable()
        prof.dump_stats(str(args.output/'cpu.prof'))
        with (args.output/'cpu_profile.txt').open('w') as stream:
            pstats.Stats(prof, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
        stages = []
        for (filename,line,function), (primitive,calls,self_time,cumulative,callers) in pstats.Stats(prof).stats.items():
            if 'doctr' in filename or function in ('<method \'cpu\' of \'torch._C.TensorBase\' objects>',):
                stages.append({'file':filename,'line':line,'function':function,'calls':calls,'self_seconds':self_time,'cumulative_seconds':cumulative})
        (args.output/'stages.json').write_text(json.dumps(sorted(stages,key=lambda s:s['cumulative_seconds'],reverse=True),indent=2),encoding='utf-8')
        with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
            model(inputs[:chunk_size])
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(args.output/'trace.json'))
        (args.output/'torch_profile.txt').write_text(trace.key_averages().table(sort_by='self_cuda_time_total', row_limit=60), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'throughput': result['throughput_per_second'], 'accuracy': accuracy}), flush=True)


if __name__ == '__main__':
    main()
