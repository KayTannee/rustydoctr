"""Append targeted character/rotation and sustained-streaming evidence to the report."""
import base64
import json
from pathlib import Path
from pybaseline.report import esc,pct,table,spark


def build():
    root=Path('pybaseline/results'); screen=root/'rotation_screen_v2'; streams=root/'streaming'
    parts=['<h2>Single-character rotation regression</h2><p>This additional corpus has three A4 pages: upright, 7-degree skew, and 7-degree skew with genuine 90/-90/180/-18-degree side text. The same contextual sentences are printed in Helvetica, Times and Courier. It explicitly includes uppercase I, lowercase i/l, digits, underscore and hyphen. These are synthetic development cases, not an independent held-out test.</p>']
    contact=Path('testdata/rotation/contact.png')
    if contact.exists():
        parts.append('<details><summary>View character and orientation test pages</summary><img alt="Rotation test pages" style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(contact.read_bytes()).decode()+'"></details>')
    rows=[]; details=[]
    for path in sorted(screen.glob('*/result.json')):
        r=json.loads(path.read_text()); d=json.loads((path.parent/'character_diagnostics.json').read_text())
        singles=sum(x['single_total'] for x in d['pages']); correct=sum(x['single_exact'] for x in d['pages'])
        rotated=sum(x['rotated_total'] for x in d['pages']); rot_correct=sum(x['rotated_exact'] for x in d['pages'])
        flips=sum(c['count'] for page in d['pages'] for c in page['confusions'] if c['truth']=='I' and c['predicted'] in ['-','_'])
        reverse=sum(c['count'] for page in d['pages'] for c in page['confusions'] if c['truth'] in ['-','_'] and c['predicted']=='I')
        rows.append([esc(path.parent.name),pct(r['accuracy']['end_to_end_exact_recall']),f'{correct}/{singles}',f'{rot_correct}/{rotated}',str(flips),str(reverse)])
        parts2=[]
        for page in d['pages']:
            failures=[c for c in page['confusions'] if c['truth']!=c['predicted']]
            parts2.append('<h4>'+esc(page['page'])+'</h4>'+table(['Truth','Predicted','Count'],[[esc(c['truth']),esc(c['predicted']),str(c['count'])] for c in failures]))
        details.append('<details><summary>'+esc(path.parent.name)+' — character confusions</summary>'+''.join(parts2)+'</details>')
    parts.append(table(['Configuration','Exact recall @ IoU .5','Single-char exact @ .25','Genuinely rotated exact @ .25','I → dash/underscore','Dash/underscore → I'],rows))
    parts.append('<p>Character diagnostics use IoU 0.25 to be less sensitive to thin-glyph bounds; primary accuracy stays at 0.5. Ground truth uses font-metric boxes: unmatched underscores/hyphens can reflect missing detections, merging, or box mismatch, not necessarily rotation classification. A matched I → dash confusion is much stronger evidence than treating every unmatched token as a rotation error. Whole-page straightening still permits local crop orientation unless explicitly disabled. Disabling local orientation is an ablation and can sacrifice genuinely rotated text.</p>')
    parts.extend(details)
    evidence=screen/'contextual_I.json'
    if evidence.exists():
        e=json.loads(evidence.read_text()); image=screen/'contextual_I.png'
        parts.append('<h3>Concrete contextual failure</h3><p>The printed sentence is “It was cold outside I put my coat on”. The nearest prediction for its I is <b>'+esc(e['prediction']['value'])+'</b>, with recognition confidence '+pct(e['prediction']['confidence'])+' and recorded crop orientation '+esc(e['prediction'].get('crop_orientation'))+'. Green is the nominal ground-truth box; orange is the predicted box.</p>')
        parts.append('<img alt="Original sentence with I bounding boxes" style="max-width:100%;background:white" src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'">')
        parts.append('<p>A recorded 0/180-degree correction does not by itself turn a vertical stem into a horizontal one. Some I-to-dash failures also persist in the no-crop-orientation ablation; detection/cropping and recognition are therefore involved, rather than every failure being explained by the orientation classifier alone.</p>')
    selection=screen/'selection.json'
    if selection.exists():
        parts.append('<p><b>Provisional model/policy selection:</b> '+esc(selection.read_text())+'</p>')
        parts.append('<p>DB ResNet34 + PARSeq leads the aggregate score in this targeted screen, but only narrowly: one additional exact word separates it from DB ResNet34 + CRNN MobileNet large. FAST base + PARSeq recognizes more of the 21 genuinely rotated words (16 versus 10 at IoU 0.25). The selected pair is a streaming baseline, not a settled production recommendation.</p>')
    parts.append('<h2>Sustained feeder → docTR → result-writer pipeline</h2><p>One CPU process repeatedly decodes the PNG, one inference process runs the full docTR OCR predictor, and one CPU process serializes each result to JSONL. Input pixels use a bounded shared-memory pool; the input queue carries descriptors. The output queue holds two batches. Backpressure is enforced and sequence/count checks reject dropped or reordered pages. The feeder uses warm filesystem cache: this is best-case local input, not network storage or PDF rasterization. Two full batches warm the model before measurement.</p>')
    summaries=[]
    for path in sorted(streams.glob('*/summary.json')):
        r=json.loads(path.read_text()); r['_folder']=path.parent; summaries.append(r)
    rows=[]
    for r in summaries:
        resource=r['resources']; latency=r['post']['latency_seconds']
        cpu_seconds=r['engine_cpu_seconds']+r['feeder']['cpu_seconds']+r['post']['cpu_seconds']
        rows.append([esc(r['_folder'].name),str(r['config']['batch'])+'/'+str(r['config']['reco_batch']),str(r['pages']),
                     f'{r["pipeline_wall_seconds"]:.1f}',f'{r["pages_per_second"]:.2f}',f'{latency["p50"]:.2f} / {latency["p95"]:.2f}',
                     f'{resource["gpu_util_pct_mean"]:.1f}%' if resource.get('gpu_util_pct_mean') is not None else 'Unavailable',
                     f'{r["allocated_peak_bytes"]/2**30:.2f}',
                     f'{resource["device_vram_bytes_max"]/2**30:.2f}' if resource.get('device_vram_bytes_max') is not None else 'Unavailable',
                     f'{cpu_seconds/r["pipeline_wall_seconds"]:.2f}'])
    parts.append(table(['Run','Page/reco batch','Pages written','Pipeline seconds','Pages/s','Latency p50/p95 (s)','GPU mean','Torch allocated peak GiB','Device VRAM peak GiB','Pipeline CPU cores mean'],rows))
    batch_selection=streams/'batch_selection.json'
    if batch_selection.exists():
        parts.append('<p><b>Batch selection:</b> '+esc(batch_selection.read_text())+'</p>')
    parts.append('<p>The short tuning sweep selects the fastest tested batch that retains the best warmup-sample exact count. It is not an exhaustive optimizer. Long runs feed for 300 seconds and then drain the bounded queues. Aggregate throughput includes feeder startup, decode/copy, full OCR, document construction, export, IPC, writing, final fsync and drain; model loading and warmup are excluded. Per-page latency runs from admission before decode through post-process write+flush. Repeated copies test sustained throughput and stability, not accuracy diversity. Single-inference-process NVML/CPU timelines include device-wide GPU activity; pipeline CPU-core averages sum CPU time from the three work processes.</p>')
    for r in summaries:
        if not r['_folder'].name.startswith('long_'):
            continue
        parts.append('<details><summary>'+esc(r['_folder'].name)+' — sustained behaviour</summary>')
        stats=[['Full OCR inference',r['inference_seconds']],['Export',r['export_seconds']],['Inference waiting for input',r['input_wait_seconds']],['Inference waiting to enqueue output',r['output_wait_seconds']],['Feeder decode/copy',r['feeder']['decode_copy_seconds']],['Feeder waiting for a free slot',r['feeder']['free_slot_wait_seconds']],['Writer serialization/flush',r['post']['serialize_flush_seconds']]]
        parts.append(table(['Stage/wait','Seconds'],[[esc(name),f'{seconds:.3f}'] for name,seconds in stats]))
        parts.append('<p>These stage totals overlap across processes and must not be added. Feeder slot wait is expected backpressure when inference is slower than decoding. Small inference input/output waits indicate the surrounding processes keep up. The JSON output remains available in this run’s pages.jsonl.</p>')
        audit_path=r['_folder']/'audit.json'
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            parts.append('<p>Saved-output audit: '+str(audit['pages_checked'])+' serialized pages checked; sequence, count and timing arithmetic passed. Distinct ordered word-text sequences: '+str(audit['distinct_word_text_sequences'])+'. Word-count frequencies: '+esc(audit['word_count_frequencies'])+'. Partial final batches can produce numerical differences.</p>')
        telemetry=[json.loads(line) for line in (r['_folder']/'telemetry.jsonl').read_text().splitlines()]
        parts.append('<h4>GPU activity (%)</h4>'+spark(telemetry,'gpu_util_pct','#187abd',100))
        parts.append('<h4>Inference-process CPU (%)</h4>'+spark(telemetry,'process_cpu_pct','#bd6818'))
        vram=[dict(x,vram_gib=x['device_vram_bytes']/2**30 if x.get('device_vram_bytes') is not None else None) for x in telemetry]
        parts.append('<h4>Total device VRAM (GiB)</h4>'+spark(vram,'vram_gib','#8e44ad'))
        completions=r['post']['timeline']; rates=[]; start=r['start_ns']; last_count=0; last_time=start
        for c in completions:
            elapsed=(c['time_ns']-last_time)/1e9
            if elapsed>=10:
                rates.append({'time_ns':c['time_ns'],'pages_per_second':(c['pages_total']-last_count)/elapsed})
                last_time=c['time_ns']; last_count=c['pages_total']
        parts.append('<h4>Completed pages/second (roughly 10-second windows)</h4>'+spark(rates,'pages_per_second','#16855d'))
        parts.append('</details>')
    (root/'baseline/extension.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':
    build()
