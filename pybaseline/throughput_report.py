"""Audit sustained throughput outputs and build the comparison report."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from pybaseline.metrics import score_words, detection_summary


def table(headers, rows):
    def cells(items, tag):
        return ''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in items)
    return '<div class="scroll"><table><tr>'+cells(headers, 'th')+'</tr>'+''.join('<tr>'+cells(r, 'td')+'</tr>' for r in rows)+'</table></div>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('pybaseline/results/throughput'))
    a=p.parse_args()
    workload=json.loads(Path('testdata/throughput.json').read_text())['pages']
    corpus_hash=hashlib.sha256(Path('testdata/throughput.json').read_bytes()).hexdigest()
    model_meta=json.loads(Path('models/metadata.json').read_text())
    for model in ['db_resnet34','parseq']:
        assert hashlib.sha256(Path(f'models/{model}.onnx').read_bytes()).hexdigest()==model_meta[model]['sha256']
    runs={}; audit={}; quality=[]; series={}
    for path in a.runs:
        name='target_4gb' if path.name=='run' and path.parent.name=='target_4gb' else path.name;r=json.loads((path/'summary.json').read_text());runs[name]=r
        assert r['config']['size']==1536 and len(r['first_pages'])==len(workload)
        if 'corpus_sha256' in r: assert r['corpus_sha256']==corpus_hash
        else: assert r['workload']==[[p['id'],p['image']] for p in workload]
        pages=words=0;unique_counts={};text_variants={}
        with (path/'pages.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);assert row['sequence']==pages and row['page']==pages%len(workload)
                pages+=1;words+=len(row['words']);unique_counts.setdefault(row['page'],set()).add(len(row['words']))
                signature=hashlib.sha256(json.dumps([w['text'] for w in row['words']],ensure_ascii=False).encode()).hexdigest()
                text_variants.setdefault(row['page'],set()).add(signature)
        assert pages==r['pages'] and pages%len(workload)==0
        if 'max_inflight' in r: assert r['max_inflight']<=r['config']['inflight']
        scores=[]
        for i,truth in enumerate(workload):
            score=score_words(truth['words'],r['first_pages'][str(i)]['words']);scores.append(score)
            quality.append([name,truth['id'],score['truth'],score['predicted'],score['matched'],score['exact']])
        telemetry=[json.loads(line) for line in (path/'telemetry.jsonl').read_text().splitlines()]
        telemetry=[x for x in telemetry if r['start_ns']<=x['time_ns']<=r['end_ns']]
        series[name]=[]
        for second in range(int(r['wall_seconds'])+1):
            bucket=[x for x in telemetry if second<=(x['time_ns']-r['start_ns'])/1e9<second+1]
            if bucket:
                series[name].append((second,{key:float(np.mean([x[key] for x in bucket if x.get(key) is not None])) for key in ['gpu_util_pct','device_vram_bytes']}))
        resources={}
        for key in ['gpu_util_pct','device_vram_bytes','rss_bytes','process_cpu_pct']:
            vals=[x[key] for x in telemetry if x.get(key) is not None]
            resources[key]={'mean':float(np.mean(vals)), 'max':max(vals), 'p10':float(np.percentile(vals,10))} if vals else {}
        quarters=[]
        for q in range(4):
            start=r['start_ns']+(r['end_ns']-r['start_ns'])*q/4;end=r['start_ns']+(r['end_ns']-r['start_ns'])*(q+1)/4
            rows=[x for x in telemetry if start<=x['time_ns']<end]
            quarters.append({key:float(np.mean([x[key] for x in rows if x.get(key) is not None])) for key in ['device_vram_bytes','rss_bytes']})
        timeline=[(x['elapsed'],x['pages']) if isinstance(x,dict) else x for x in r['timeline']]
        windows=[]
        for end in range(60,int(r['wall_seconds'])+1,60):
            before=max((c for t,c in timeline if t<=end-60),default=0);after=max((c for t,c in timeline if t<=end),default=0)
            windows.append((after-before)/60)
        audit[name]={'pages':pages,'words':words,'wall_seconds':r['wall_seconds'],'pages_per_second':r['pages_per_second'],
                     'resources':resources,'memory_quarters':quarters,'minute_pages_per_second':windows,
                     'word_counts_per_page':{k:sorted(v) for k,v in unique_counts.items()},'text_variants_per_page':{k:len(v) for k,v in text_variants.items()},'accuracy':detection_summary(scores)}
    native_names=[n for n,r in runs.items() if r['implementation'].startswith('rust')]
    refs=[n for n,r in runs.items() if r['implementation'].startswith('python')]
    parity=[]
    for name in native_names:
        for ref in refs:
            counts=[]
            for i,truth in enumerate(workload):
                s=score_words(runs[ref]['first_pages'][str(i)]['words'],runs[name]['first_pages'][str(i)]['words'],.99)
                counts.append(s)
                parity.append(dict(rust=name,reference=ref,page=truth['id'],**s))
            audit[name].setdefault('parity',{})[ref]=detection_summary(counts)
    result={'runs':audit,'strict_parity':parity,'corpus_sha256':corpus_hash,'models':model_meta}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Rust OCR throughput</title>
    <style>body{font:16px system-ui;margin:36px auto;padding:0 24px;max-width:1280px;color:#203047;background:#f4f6fa}p{line-height:1.6}h2{margin-top:36px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #d9e0e8}th{background:#243b56;color:white}code{background:#e4eaf2;padding:2px 5px}svg{width:100%;max-height:280px}a{color:#146dc2}</style>
    <h1>Rust OCR: sustained throughput</h1>
    <p>DB ResNet34 + PARSeq, detector 1536 × 1536, FP32, TF32 disabled. Ten upright synthetic pages, including dense A3, repeated in a fixed order. Each corpus pass contains 16,246 labelled words. This pair is the provisional quality choice from the earlier model/rotation studies; this experiment changes throughput, not the model selection.</p>
    <p>Warm-cache PNG decode → detector → word geometry/crops (including wide-crop split/remap) → recognizer → ordered JSONL. Loading and a complete-corpus warmup are excluded; admission, queue drain, decode, inference and output flush/fsync are timed. PDF rasterization, orientation, retries and document lines/blocks are excluded. Runs execute sequentially on the RTX 5070 Ti 16 GB / Ryzen 7 8700G. Repeated synthetic pages are a controlled workload, not a production scan benchmark.</p>''']
    if 'python_binding' in runs and 'native' in runs:
        parts.append('<p>The final Python-library and native rows are a fresh pair measured after a reboot; the other rows are earlier runs. Their 0.4% difference shows no measurable interface penalty in these single runs. The 4 GB target row includes 2.21 GiB of initial desktop VRAM: incremental peak was 3.37 GiB.</p>')
    rows=[]
    for name,r in runs.items():
        s=audit[name];m=s['resources'];rows.append([name,f"{s['pages_per_second']:.3f}",s['pages'],f"{s['wall_seconds']:.1f}",
        f"{m['gpu_util_pct']['mean']:.1f}%",f"{m['device_vram_bytes']['max']/2**30:.2f}",f"{m['rss_bytes']['max']/2**30:.2f}",f"{m['process_cpu_pct']['mean']:.1f}%"])
    parts.append(table(['Run','Pages/s','Pages','Seconds','Mean GPU','Peak VRAM GiB','Peak RSS GiB','Mean CPU'],rows))
    for native in native_names:
        parts.append('<p>'+html.escape(native)+': '+', '.join(f"<b>{runs[native]['pages_per_second']/runs[ref]['pages_per_second']:.2f}×</b> {html.escape(ref)}" for ref in refs)+'.</p>')
    if 'sustained_rust' in runs and 'rust_one_page' in runs:
        gain=runs['sustained_rust']['pages_per_second']/runs['rust_one_page']['pages_per_second']-1
        parts.append(f'<p>Four admitted pages improved Rust throughput by <b>{gain:.1%}</b> over one admitted page. This measures cross-page overlap and batching together; the one-page case still overlaps work within a page.</p>')
    if 'balanced_rust' in runs and 'sustained_rust' in runs:
        retained=runs['balanced_rust']['pages_per_second']/runs['sustained_rust']['pages_per_second']
        saved=1-audit['balanced_rust']['resources']['device_vram_bytes']['max']/audit['sustained_rust']['resources']['device_vram_bytes']['max']
        parts.append(f'<p>The 512-crop configuration retained <b>{retained:.1%}</b> of maximum measured throughput with <b>{saved:.1%} less peak device-wide VRAM</b>. This is the preferred memory/performance trade-off on this workstation.</p>')
    parts.append('<p>Device-wide VRAM includes the desktop and other GPU allocations. CPU 100% means one logical core. NVML samples are polled every 50 ms (Python) / 100 ms (Rust); the sensor updates more slowly. Utilization measures busy time, not useful work per second. These are individual sustained runs, not statistical confidence intervals.</p>')
    colours=['#1769aa','#ba4c12','#14804a','#8d42aa','#c08a00','#d12771','#008c95','#4b5563']
    for key,title,ceiling,divisor in [('gpu_util_pct','GPU busy time (%)',100,1),('device_vram_bytes','Device-wide VRAM (GiB)',16,2**30)]:
        duration=max(r['wall_seconds'] for r in runs.values());svg=[]
        for v in [0,ceiling/2,ceiling]:
            y=230-v/ceiling*200
            svg.append(f'<line x1="55" y1="{y}" x2="1130" y2="{y}" stroke="#ccd5df"/><text x="12" y="{y+5}" font-size="14">{v:g}</text>')
        for (name,values),colour in zip(series.items(),colours):
            points=' '.join(f'{55+t/duration*1075:.1f},{230-v[key]/divisor/ceiling*200:.1f}' for t,v in values)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for second in range(0,int(duration)+1,60):
            svg.append(f'<text x="{55+second/duration*1075}" y="255" text-anchor="middle" font-size="14">{second}s</text>')
        parts.append(f'<h2>{title}</h2><svg role="img" aria-label="{title} during measured runs, one-second means" viewBox="0 0 1160 270">'+''.join(svg)+'</svg><p>'+', '.join(f'<span style="color:{colour}">{html.escape(name)}</span>' for name,colour in zip(series,colours))+' — one-second means.</p>')
    parts.append('<h2>Batching and bounded memory</h2>'+table(['Run','Detector batch','Recognition batch','Page slots','CPU post workers'],[[n,r['config'].get('det_batch',r['config'].get('page_batch')),r['config']['reco_batch'],r.get('max_inflight','2 input batches + active processing'),r['config'].get('workers','docTR/OpenCV')] for n,r in runs.items()]))
    parts.append('<p>Rust holds a page admission credit from decode until ordered output is flushed. Prepared-page, detector-result and crop-chunk queues each hold at most two items; recognition combines crops across admitted pages and flushes partial batches after a short idle interval. Two independent model sessions permit CPU stages and GPU submissions to overlap. No model-per-worker duplication.</p><p>Startup tuning tries several detector/recognizer batch sizes in fresh processes, requires identical calibration text to its smallest successful candidate and checks sampled memory growth. Automatic arena budget: 60% of initially free device memory, capped at 8 GiB; split 25% detector / 75% recognizer. Page slots derive from a conservative host-memory estimate (default 512 MiB, maximum eight slots). Arena limits are not a hard total-VRAM cap; sampled peaks can miss transients, and the host estimate excludes runtime overhead. This is startup calibration, not continuous adaptation or an OOM guarantee.</p>')
    calibration=a.output/'rust_calibration/autotune.json'
    if calibration.exists():
        tuning=json.loads(calibration.read_text());trials=[]
        for t in tuning['trials']:
            c=t.get('config',t);path=calibration.parent/f"calibrate_d{c['det_batch']}_r{c['reco_batch']}"/'summary.json'
            summary=json.loads(path.read_text()) if path.exists() else None
            trials.append([c['det_batch'],c['reco_batch'],f"{t['pages_per_second']:.3f}" if summary else 'failed',
                           f"{summary['resources']['device_vram_bytes_max']/2**30:.2f}" if summary else 'arena limit',
                           t.get('same_calibration_text','—'),t.get('within_observed_budget','—')])
        parts.append('<h2>Startup calibration trade-offs</h2>'+table(['Detector batch','Recognition batch','Pages/s (10 pages)','Peak device VRAM GiB','Same text','Within observed budget'],trials))
        parts.append('<p>The larger detector batch exceeded its assigned arena and the tuner recovered by rejecting that trial. This initial calibration used fastest-only selection. The current default chooses the least-memory eligible trial within 3% of the fastest (512 crops on these measurements); use --throughput-tolerance 0 for fastest-only selection. Short-trial memory is not a guarantee, because runtime caches can grow later and arena caps exclude some allocations.</p>')
        for n in native_names:
            increment=runs[n]['resources'].get('vram_increment_peak_bytes')
            budget=runs[n]['config']['arena_mib']*2**20
            if increment and increment>budget:
                parts.append(f'<p>{html.escape(n)} reached {increment/2**30:.2f} GiB incremental VRAM, beyond its {budget/2**30:g} GiB short-trial target. This is an observed limitation of the startup memory gate, not a hard total-memory guarantee.</p>')
    parts.append('<h2>Stability</h2>'+table(['Run','Full-minute pages/s','VRAM quarter means GiB','RSS quarter means GiB'],[[n,', '.join(f'{x:.3f}' for x in s['minute_pages_per_second']),', '.join(f"{x['device_vram_bytes']/2**30:.2f}" for x in s['memory_quarters']),', '.join(f"{x['rss_bytes']/2**30:.2f}" for x in s['memory_quarters'])] for n,s in audit.items()]))
    parts.append('<p>Every JSONL record was checked for ordered, contiguous sequence numbers and the expected corpus index. All admitted pages completed. Minute rates include uneven page costs and batch completion boundaries. Raw telemetry and per-page predictions are retained beside each summary.</p>')
    parts.append(table(['Run','Pages with repeated-text variation'],[[n,sum(v>1 for v in s['text_variants_per_page'].values())] for n,s in audit.items()]))
    parts.append('<h2>Quality guardrails</h2>'+table(['Run','Truth','Predicted','IoU ≥ 0.5 matched','Exact text','Exact recall'],[[n,s['accuracy']['truth'],s['accuracy']['predicted'],s['accuracy']['matched'],s['accuracy']['exact'],f"{s['accuracy']['end_to_end_exact_recall']:.2%}"] for n,s in audit.items()]))
    parts.append('<p>Accuracy uses the first occurrence of each of the ten pages. Dense A3 is intentionally beyond reliable segmentation at this resolution. One-box and small-coordinate differences were accepted for this throughput phase; they are not hidden by aggregate scores.</p>')
    parts.append('<details><summary>Per-page labelled-word scores</summary>'+table(['Run','Page','Truth','Predicted','Matched boxes','Exact words'],quality)+'</details>')
    parts.append(table(['Rust run','Reference','Page','Reference boxes','Rust boxes','IoU ≥ 0.99 matches','Same matched text'],[[x['rust'],x['reference'],x['page'],x['truth'],x['predicted'],x['matched'],x['exact']] for x in parity]))
    if 'python_binding' in runs and 'native' in runs:
        ratio=runs['python_binding']['pages_per_second']/runs['native']['pages_per_second']
        parts.append(f'<h2>Installed Python library</h2><p>The installed wheel achieved <b>{ratio:.1%}</b> of the fresh native run with identical parameters. Python decodes PNGs, submits RGB bytes on one thread, and independently parses/writes results on the consumer. Neither run times model loading or full-corpus warmup. The Python boundary adds an owned RGB copy and JSON parsing; these overlap with inference. Runs are single measurements, so small differences are not evidence of a language speedup. Both were repeated after the system restart.</p>')
    parts.append('<h2>What the result proves</h2><p>The comparison measures complete implementations and scheduling choices. Python/Torch uses native docTR models and early-ending PARSeq decoding; Python/ORT and Rust use identical exported graphs, which decode the full sequence. Python overlaps input decoding but executes OCR stages serially; Rust also overlaps detection, crop preparation and recognition. The same-ORT comparison controls the model backend more closely, but does not isolate the language, GIL, batching or scheduler individually. A similarly pipelined Python implementation could recover some of the gain. PyTorch uses detector batch four and its own uncapped allocator; its VRAM difference is not solely a language effect.</p>')
    parts.append('<h2>Stage wall times</h2>'+table(['Run','Stage','Total seconds','Milliseconds/page'],[[n,k,f'{v:.2f}',f'{v/r["pages"]*1000:.1f}'] for n,r in runs.items() for k,v in r.get('stage_seconds',{}).items()]))
    parts.append('<p>Stage times overlap and must not be added. Recognition includes transfers, inference, decoding and result handoff. Queue-wait time measures backpressure, not lost GPU time. Further work should follow measured recognition/input waits and pages/s rather than chasing a 100% GPU-Z reading.</p><h2>Next CPU work and smaller GPUs</h2><p>Additional CPU checks can leave steady-state throughput unchanged while workers finish ahead of GPU demand, although per-page latency can rise. Extra inference passes still consume GPU time. The 4 GB target now uses smaller batches and an incremental VRAM budget based on actual free memory, with both FP32 models resident. Its desktop proxy run completed 290 pages at 0.928 pages/s, peaking at 3.371 GiB above the 2.215 GiB initial device usage. The actual 4 GB laptop is still untested; these results do not certify that hardware. Reprofile on the destination machine.</p><p>See <a href="comparison.json">comparison.json</a> for model hashes, all metrics and strict parity counts. Commands and VS Code tasks are documented in THROUGHPUT.md.</p></html>')
    (a.output/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps({n:{k:v for k,v in s.items() if k in ['pages_per_second','accuracy','resources']} for n,s in audit.items()},indent=2))


if __name__=='__main__':main()
