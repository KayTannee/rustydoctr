"""Build a portable HTML report from measured result.json and raw telemetry."""
import argparse
import base64
import html
import json
from pathlib import Path
import statistics


def esc(value):
    return html.escape(str(value))


def pct(value):
    return f'{100*value:.2f}%'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def spark(rows, key, color, maximum=None):
    pairs = [(r['time_ns'], r.get(key)) for r in rows if r.get(key) is not None]
    if not pairs:
        return '<p>Sensor unavailable.</p>'
    start, end = pairs[0][0], pairs[-1][0]
    top = maximum or max(v for _,v in pairs) or 1
    points = ' '.join(f'{(t-start)/max(end-start,1)*900:.1f},{130-v/top*120:.1f}' for t,v in pairs)
    return f'<svg viewBox="0 0 900 145" role="img" aria-label="{esc(key)} timeline"><path d="M0 130H900" stroke="#bbc"/><polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/></svg><small>0 to {top:.0f}; {(end-start)/1e9:.1f} seconds</small>'


def build(root, output):
    results = []
    failures = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if (folder/'result.json').exists():
            r = json.loads((folder/'result.json').read_text())
            r['_folder'] = folder
            results.append(r)
        elif (folder/'status.json').exists():
            failures.append((folder.name, json.loads((folder/'status.json').read_text())))
    if not results:
        raise ValueError('No completed measurements')
    if len({r['corpus_sha256'] for r in results}) != 1:
        raise ValueError('Cannot rank results from different corpora in one report')
    first = results[0]
    word_count = first['accuracy'].get('truth', first['accuracy'].get('words'))
    manifest_path = Path(first['config']['manifest'])
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dpi = manifest.get('dpi', 'recorded in the source manifest')
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>docTR baseline</title>',
             '<style>body{font:16px system-ui;background:#f4f6fa;color:#19283b;max-width:1400px;margin:40px auto;padding:0 24px;overflow-wrap:anywhere}pre{overflow-x:auto;white-space:pre-wrap}h1{font-size:40px}h2{margin-top:42px}p{max-width:1100px;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce3ec}th{background:#23374e;color:white}td:first-child{font-weight:600}small{color:#58687b}details{background:white;margin:12px 0;padding:18px;border-radius:8px}summary{cursor:pointer;font-weight:650}.scroll{overflow-x:auto}.callout{background:#e0ebf6;border-left:5px solid #2676b8;padding:18px}svg{width:100%;max-height:190px}code{background:#e4e9ef;padding:2px 5px}a{color:#176eaa}</style>',
             f'<h1>docTR · measured baseline</h1><p>Word detection, recognition and pipeline behaviour on {esc(first["environment"]["gpu"])}. Synthetic stress corpus, not a real-document accuracy leaderboard.</p>',
             f'<p>{len(results)} completed baseline configurations. Additional character/rotation and streaming results appear below. docTR {esc(results[0]["environment"]["doctr"])} · PyTorch {esc(results[0]["environment"]["torch"])} · CUDA {esc(results[0]["environment"]["cuda"])} · {results[0]["environment"]["logical_cpus"]} logical CPUs.</p>']
    det = [r for r in results if r['config']['kind']=='detection']
    reco = [r for r in results if r['config']['kind']=='recognition']
    ocr = [r for r in results if r['config']['kind']=='ocr']
    insights = []
    if det:
        best = max(det,key=lambda r:r['accuracy']['f1'])
        fastest = max(det,key=lambda r:r['throughput_per_second'])
        insights.append(f'Highest detector F1: <b>{esc(best["config"]["det"])}</b> ({pct(best["accuracy"]["f1"])}). Fastest detector: <b>{esc(fastest["config"]["det"])}</b> ({fastest["throughput_per_second"]:.2f} pages/s).')
    if reco:
        best = max(reco,key=lambda r:r['accuracy']['exact_accuracy'])
        insights.append(f'Highest isolated recognition accuracy: <b>{esc(best["config"]["reco"])}</b>, {pct(best["accuracy"]["exact_accuracy"])} exact words, {pct(best["accuracy"]["cer"])} CER on upright ground-truth crops.')
    baseline = next((r for r in ocr if r['_folder'].name=='ocr_fast_base_parseq'),None)
    higher = next((r for r in ocr if r['_folder'].name.endswith('_size1536')),None)
    if baseline and higher:
        insights.append(f'Increasing FAST-base input from 1024 to 1536 changes end-to-end exact word recall from {pct(baseline["accuracy"]["end_to_end_exact_recall"])} to {pct(higher["accuracy"]["end_to_end_exact_recall"])}; throughput changes from {baseline["throughput_per_second"]:.2f} to {higher["throughput_per_second"]:.2f} pages/s.')
    parts.append('<div class="callout">'+'<p>'+ '</p><p>'.join(insights)+'</p></div>')
    findings = root/'findings.html'
    if findings.exists():
        parts.append(findings.read_text(encoding='utf-8'))
    extension=root/'extension.html'
    if extension.exists():
        parts.append(extension.read_text(encoding='utf-8'))
    environment = root.parent/'environment.json'
    if environment.exists():
        parts.append('<details><summary>Hardware and driver snapshot</summary><pre>'+esc(environment.read_text())+'</pre></details>')
    preview = manifest_path.parent/'contact.png'
    if preview.exists():
        parts.append('<details><summary>View the synthetic test pages</summary><img style="max-width:100%" alt="Synthetic PDF contact sheet" src="data:image/png;base64,'+base64.b64encode(preview.read_bytes()).decode()+'"></details>')
    parts.append(f'<h2>How to interpret these results</h2><p>{first["pages"]} unique vector PDF pages; {word_count:,} labelled words. Raster DPI: {esc(dpi)}. Repetition increases timing duration, not dataset diversity. Each run uses a fresh process, float32, eval/inference mode, a full-corpus warmup and CUDA synchronization at timing boundaries. Minimum timed passes: {min(r["config"]["repeats"] for r in results)}; minimum target duration: {min(r["config"]["min_seconds"] for r in results):g} seconds. Read each configuration below for overrides. Download, model loading, PDF rasterization, image loading, crop preparation, scoring and serialization are excluded from throughput. OCR timing includes detector and recognizer preprocessing, inference, postprocessing, cropping, orientation handling when enabled, and document construction.</p>')
    parts.append('<p>Detector scores use polygon IoU ≥ 0.5 and maximum-cardinality one-to-one matching. Ground truth is based on font advance widths and ascent/descent, not tight ink bounds. Precision/recall/F1 are micro-averaged; the very dense A3 page therefore has substantial weight. End-to-end exact recall requires both a matched box and case-sensitive exact text. Recognition uses all ground-truth word crops, geometrically rectified upright, so it does not test detection or orientation classification. CER is summed Levenshtein character edits / total ground-truth characters. These Latin words and five built-in fonts are a controlled stress test, not representative English or scanned-document evidence.</p>')
    parts.append('<p>Merge candidates cover ≥50% of two or more ground-truth words. Split candidates have two or more predictions each ≥50% inside one ground-truth word. These are geometry diagnostics, not definitive semantic error counts. A single IoU threshold can penalize padded boxes. Inspect per-page results before choosing a model; lower throughput can also mean that a detector found more words for the recognizer.</p>')
    for kind, title, data in [('detection','Detector rankings',det),('recognition','Recognizer rankings',reco),('ocr','End-to-end and configuration experiments',ocr)]:
        parts.append('<h2>'+title+'</h2>')
        data = sorted(data,key=lambda r:r['accuracy'].get('exact_accuracy',r['accuracy'].get('f1',0)),reverse=True)
        headers = ['Configuration','Words/s' if kind=='recognition' else 'Pages/s','Exact / F1','CER / recall','E2E exact recall','GPU mean / max','CPU cores mean','Torch peak GiB','Device peak GiB']
        rows=[]
        for r in data:
            a, resource = r['accuracy'], r['resources']
            gpu = resource.get('gpu_util_pct_mean')
            rows.append([esc(r['_folder'].name),f'{r["throughput_per_second"]:.2f}',pct(a.get('exact_accuracy',a.get('f1',0))),pct(a.get('cer',a.get('recall',0))),pct(a['end_to_end_exact_recall']) if kind=='ocr' else '—',
                         f'{gpu:.1f}% / {resource["gpu_util_pct_max"]:.0f}%' if gpu is not None else 'Unavailable',f'{resource["process_cpu_pct_mean"]/100:.2f}',
                         f'{r["memory"]["allocated_peak_bytes"]/2**30:.2f}',f'{resource["device_vram_bytes_max"]/2**30:.2f}' if resource.get('device_vram_bytes_max') else 'Unavailable'])
        parts.append(table(headers,rows))
    parts.append('<h2>Resource monitoring and the GIL hypothesis</h2><p>The sampler is a separate process requesting readings every 50 ms. NVML GPU utilisation is the fraction of its sensor window with a kernel running, not percentage of theoretical compute throughput. Polling faster does not create new hardware samples. Memory-controller utilisation differs from VRAM allocation. Device VRAM and GPU activity include the desktop and other applications; PyTorch allocated/reserved peaks are process allocator counters and omit some CUDA context/library allocations. Windows WDDM may not expose per-process VRAM. CPU cores mean = process CPU percent / 100; 100% means one logical core.</p><p>Low GPU activity with substantial CPU use is compatible with host work and synchronization gaps, but does not prove GIL contention. Current docTR already uses thread pools for preprocessing; PyTorch/OpenCV perform native work. Separate cProfile and PyTorch traces, where provided, locate expensive calls without contaminating throughput. CUDA profiling can require CUPTI; missing GPU events must not be interpreted as zero GPU work. A Rust rewrite should follow measured stage costs and preserve numerical/geometry behaviour before changing scheduling.</p>')
    parts.append('<h2>Current upstream implementation</h2><p>docTR 1.0 removed TensorFlow and made PyTorch the sole backend; it also changed the CRNN-VGG16 checkpoint and fixed an orientation-estimation condition. Version 1.1 adds optional layout/table functionality and preservation of original coordinates after page straightening. These features are disabled here except coordinate remapping for the straightening experiment. <a href="https://github.com/mindee/doctr/releases">Release notes</a>.</p><p>Inspection of installed 1.1.0 shows FAST reparameterization enabled by the factory, threaded preprocessing, and detector maps copied back to CPU for contour processing. The OCR predictor only materializes returned segmentation maps when orientation/straightening needs them. Page straightening still invokes detection again after rotating pages. These are source observations, not a measured speedup over older docTR versions. No historical version A/B claim is made. <a href="https://github.com/mindee/doctr/tree/v1.1.0/doctr/models">Tagged model source</a>.</p>')
    parts.append('<h2>Per-configuration evidence</h2>')
    parts.append('<details><summary>Concrete changes since docTR 0.12 (source comparison)</summary><p>Compared with tagged 0.12.0, version 1.1.0 avoids unconditional OCR segmentation-map materialization, uses local rather than full-page masks for rotated-box scoring, parallelizes detector postprocessing across samples, and caps thread-pool workers to the item count while skipping single-item pool startup. The rotated-score arithmetic also changes for zero-valued probabilities. These are verified code differences, not a quantified historical speedup. Page straightening still runs detection twice.</p><p><a href="https://github.com/mindee/doctr/blob/v0.12.0/doctr/models/detection/core.py">Old detector postprocessor</a> · <a href="https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/detection/core.py">New detector postprocessor</a> · <a href="https://github.com/mindee/doctr/blob/v0.12.0/doctr/models/predictor/pytorch.py">Old OCR predictor</a> · <a href="https://github.com/mindee/doctr/blob/v0.12.0/doctr/utils/multithreading.py">Old thread helper</a>.</p></details>')
    for r in results:
        folder = r['_folder']
        parts.append(f'<details><summary>{esc(folder.name)} · {r["throughput_per_second"]:.2f} items/s</summary>')
        parts.append(f'<p>{len(r["trials_seconds"])} timed passes, {sum(r["trials_seconds"]):.1f}s total. Corpus-pass median {r["trial_seconds_median"]:.3f}s; p95 {r["trial_seconds_p95"]:.3f}s. These are corpus-pass latencies, not page latencies. Warmup {r["warmup_seconds"]:.2f}s. Model load including any download {r["model_load_seconds_including_download"]:.2f}s.</p>')
        pp = []
        for p in r['by_page']:
            if r['config']['kind']=='recognition':
                pp.append([esc(p['page']),str(p['words']),pct(p['exact_accuracy']),pct(p['cer'])])
            else:
                pp.append([esc(p['page']),str(p['truth']),str(p['predicted']),pct(p['matched']/max(p['predicted'],1)),pct(p['matched']/max(p['truth'],1)),pct(p['exact']/max(p['truth'],1)) if r['config']['kind']=='ocr' else '—',str(p['merge_candidates']),str(p['split_candidates'])])
        parts.append(table(['Page','Words','Exact accuracy','CER'] if r['config']['kind']=='recognition' else ['Page','Truth words','Predicted boxes','Precision','Detection recall','Exact recall','Merge candidates','Split candidates'],pp))
        telemetry = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
        parts.append('<h3>GPU utilisation (%)</h3>'+spark(telemetry,'gpu_util_pct','#187abd',100))
        parts.append('<h3>Process CPU (%; 100 = one core)</h3>'+spark(telemetry,'process_cpu_pct','#bd6818'))
        vram_rows = [dict(row, device_vram_gib=row['device_vram_bytes']/2**30 if row.get('device_vram_bytes') is not None else None) for row in telemetry]
        parts.append('<h3>Total device VRAM (GiB)</h3>'+spark(vram_rows,'device_vram_gib','#8e44ad'))
        parts.append('<h3>Configuration</h3><pre>'+esc(json.dumps(r['config'],indent=2))+'</pre></details>')
    if failures:
        parts.append('<h2>Failed or unavailable cases</h2>'+table(['Case','Status'],[[esc(n),esc(s)] for n,s in failures]))
    parts.append('<p>Sensor definitions: <a href="https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html">NVIDIA NVML documentation</a> describes utilisation windows of roughly one-sixth to one second depending on the device. Use a CUDA timeline profiler for individual kernels and transfer gaps.</p>')
    parts.append('<p><small>Raw result.json, predictions.json, telemetry.jsonl and run.log accompany each configuration. The report embeds its tables and plots and works offline.</small></p></html>')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(parts),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('pybaseline/results/baseline'))
    p.add_argument('--output',type=Path,default=Path('pybaseline/results/report.html'))
    a=p.parse_args(); build(a.input,a.output)
