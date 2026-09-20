"""Build measured comparisons from the baseline, profile and concurrency runs."""
import json
from pathlib import Path
from pybaseline.report import esc, pct, table


def build(root=Path('pybaseline/results/baseline')):
    results={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/result.json')}
    base=results.get('ocr_fast_base_parseq')
    parts=['<h2>Measured pipeline findings</h2>']
    recognizers=[r for r in results.values() if r['config']['kind']=='recognition']
    if recognizers:
        lo=min(recognizers,key=lambda r:r['accuracy']['exact_accuracy'])
        hi=max(recognizers,key=lambda r:r['accuracy']['exact_accuracy'])
        parts.append(f'<p>Isolated recognizer exact accuracy ranges from {pct(lo["accuracy"]["exact_accuracy"])} ({esc(lo["config"]["reco"])}) to {pct(hi["accuracy"]["exact_accuracy"])} ({esc(hi["config"]["reco"])}). Clean, rectified crops make most models look very strong; SAR and MASTER are notable exceptions in this run. Fixed two-pixel crop context and synthetic typography can affect architectures differently, and dense small text dominates the word-weighted score.</p>')
    names=['ocr_fast_base_parseq','ocr_fast_base_parseq_size1536','ocr_fast_base_parseq_stretch',
           'ocr_fast_base_parseq_batch4','ocr_fast_base_parseq_threads1',
           'ocr_fast_base_parseq_rotation_crops','ocr_fast_base_parseq_rotation_straighten']
    rows=[]
    for name in names:
        if name not in results:
            continue
        r=results[name]
        pages={p['page']:p for p in r['by_page']}
        a3=pages['a3_very_dense']; mixed=pages['a4_mixed_rotation']
        rows.append([esc(name.removeprefix('ocr_fast_base_parseq') or 'reference'),
                     f'{r["throughput_per_second"]:.2f}',pct(r['accuracy']['end_to_end_exact_recall']),
                     pct(a3['matched']/a3['truth']),pct(mixed['exact']/mixed['truth']),
                     f'{r["memory"]["allocated_peak_bytes"]/2**30:.2f}'])
    parts.append(table(['FAST-base/PARSeq variant','Pages/s','All-page exact recall','Dense A3 detection recall','Mixed-rotation exact recall','Torch peak GiB'],rows))
    sensitivity=root/'iou_sensitivity.json'
    if sensitivity.exists():
        rows=[]
        for r in json.loads(sensitivity.read_text()):
            a3=next(p for p in r['by_page'] if p['page']=='a3_very_dense')
            rows.append([esc(r['case']),str(r['iou_threshold']),pct(r['accuracy']['recall']),
                         pct(r['accuracy']['end_to_end_exact_recall']),pct(a3['matched']/a3['truth'])])
        parts.append('<h3>Localization threshold sensitivity</h3>'+table(['Case','IoU threshold','All-page box recall','All-page exact recall','Dense A3 box recall'],rows))
        parts.append('<p>This rescores the same saved predictions without rerunning models. A large increase at IoU 0.25 indicates approximate detections rejected by the stricter boundary test. Low IoU-0.5 recall should not automatically be read as an equally large percentage of wholly absent detections. The official tables above retain IoU 0.5.</p>')
    if base and 'ocr_fast_base_parseq_threads1' in results:
        r=results['ocr_fast_base_parseq_threads1']
        ratio=r['throughput_per_second']/base['throughput_per_second']
        parts.append(f'<p>Setting PyTorch/OpenCV intra-op threads from 4 to 1 changes throughput by <b>{ratio:.2f}×</b>. docTR’s preprocessing/postprocessing thread pools remain enabled. This tests native thread scheduling/oversubscription as well as host work; it is not a measurement of the GIL alone.</p>')
    parts.append('<h3>One versus two processes</h3>')
    concurrency=[]
    for workers in [1,2]:
        path=root.parent/f'concurrency{workers}/summary.json'
        if path.exists():
            concurrency.append(json.loads(path.read_text()))
    parts.append(table(['Workers','Aggregate pages/s','Device peak GiB','Summed Torch allocated peaks GiB','GPU average by sampler','Start spread (s)'],[
        [str(r['workers']),f'{r["aggregate_pages_per_second"]:.2f}',f'{r["device_peak_bytes"]/2**30:.2f}',
         f'{r["torch_peak_allocated_sum_bytes"]/2**30:.2f}',esc([round(x,1) if x is not None else None for x in r['gpu_mean_pct_per_worker']]),f'{r["start_spread_seconds"]:.3f}'] for r in concurrency]))
    if len(concurrency)==2:
        speed=concurrency[1]['aggregate_pages_per_second']/concurrency[0]['aggregate_pages_per_second']
        parts.append(f'<p>Two independent workers deliver <b>{speed:.2f}×</b> aggregate throughput versus one. Both configurations use four intra-op threads and identical batch sizes. Device peaks include the desktop; summed Torch peaks are upper bounds on concurrent allocated memory, not a directly sampled sum. Runs are sequential experiments on a shared desktop, not a controlled production-load test.</p>')
    profile=root.parent/'profile/stages.json'
    if profile.exists():
        stages=json.loads(profile.read_text())
        selected=[]
        for s in stages:
            file=s['file'].replace('\\','/')
            if (s['function'] in ['forward','__call__'] and any(part in file for part in ['/models/preprocessor/','/models/detection/predictor/','/models/recognition/predictor/','/models/predictor/'])) or (s['function'] in ['_prepare_crops','_rectify_crops','__call__'] and ('/models/predictor/' in file or '/models/builder' in file)):
                selected.append([esc(file.split('/doctr/')[-1]+':'+s['function']),str(s['calls']),f'{s["cumulative_seconds"]:.3f}',f'{s["self_seconds"]:.3f}'])
        parts.append('<h3>Separate full-corpus cProfile diagnostic</h3>'+table(['Stage','Calls','Cumulative seconds','Self seconds'],selected))
        parts.append('<p>These diagnostic times include profiler overhead. Cumulative stage times overlap and must not be summed; preprocessing is nested inside detection and recognition. Threaded work, native calls and waits complicate cProfile attribution, so this is not direct evidence of GIL contention. The separate PyTorch trace captured real CUDA events and is the appropriate artifact for examining kernel and transfer gaps in the profiled two-page batch.</p>')
    parts.append('<h3>What this establishes before Rust work</h3><p>The synthetic corpus isolates severe small-text/downsampling failures from recognizer capability. Benchmark a native implementation against the saved word polygons and predictions, while treating changes to resolution, tiling, orientation and scheduling as separate experiments. These runs do not quantify the speedup a Rust rewrite would achieve, and clean synthetic text cannot select the best recognizer for real scans. Real problematic pages and stage-level equivalence checks are the next evidence needed before committing to an implementation design.</p>')
    (root/'findings.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':
    build()
