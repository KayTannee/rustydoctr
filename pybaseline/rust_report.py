"""Audit the native word slice against Python/ORT and Python/PyTorch."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pybaseline.metrics import score_words,detection_summary
from pybaseline.report import table,esc,pct


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    root=Path('pybaseline/results/rust_slice')
    results={name:read(root/name/'result.json') for name in ['rust','python_ort','python_torch']}
    manifest=read(Path('testdata/generated/manifest.json'))
    truth={p['image']:p for p in manifest['pages']}
    all_rows=[];audit={};all_scores={}
    for name,r in results.items():
        assert r['size']==1024 and r['reco_batch']==128
        assert len(r['trials_seconds'])>=3 and sum(r['trials_seconds'])>=30
        assert len(r['pages'])==3
        score=[]
        for page in r['pages']:
            label=truth[Path(page['image']).name]
            s=score_words(label['words'],page['words']);score.append(s)
            all_rows.append([name,esc(label['id']),str(s['truth']),str(s['predicted']),str(s['matched']),str(s['exact'])])
        all_scores[name]=detection_summary(score)
        audit[name]={'pages_per_second':r['pages_per_second'],'timed_seconds':sum(r['trials_seconds']),
                     'trials':len(r['trials_seconds']),'accuracy':all_scores[name]}
    parity=[]
    for name in ['python_ort','python_torch']:
        for native,ref in zip(results['rust']['pages'],results[name]['pages']):
            assert Path(native['image']).name==Path(ref['image']).name
            s=score_words(ref['words'],native['words'],threshold=.99,return_pairs=True)
            differences=[{'reference':ref['words'][i], 'rust':native['words'][j]} for i,j in s['pairs'] if ref['words'][i]['text']!=native['words'][j]['text']]
            max_delta=max((float(np.max(np.abs(np.asarray(ref['words'][i]['polygon'])-native['words'][j]['polygon']))) for i,j in s['pairs']),default=None)
            parity.append(dict(reference=name,page=Path(native['image']).stem,
                               **{k:v for k,v in s.items() if k!='pairs'},max_matched_coordinate_delta=max_delta,text_differences=differences))
    numerical={}
    for name in ['detector_input','detector_probability','recognizer_input']:
        a=np.fromfile(root/'rust'/f'{name}.f32',np.float32)
        b=np.fromfile(root/'python_ort'/f'{name}.f32',np.float32)
        if a.shape==b.shape:
            numerical[name]=dict(elements=a.size,max_abs=float(np.abs(a-b).max()),mean_abs=float(np.abs(a-b).mean()),identical_fraction=float(np.mean(a==b)))
    metadata=read(Path('models/metadata.json'))
    for name in ['db_resnet34','parseq']:
        assert hashlib.sha256((Path('models')/f'{name}.onnx').read_bytes()).hexdigest()==metadata[name]['sha256']
    audit.update(parity=parity,numerical=numerical,models=metadata)
    (root/'comparison.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Rust OCR vertical slice</title>',
           '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 24px;background:#f4f6fa;color:#19283b;overflow-wrap:anywhere}p{line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd}th{background:#23374e;color:white}h2{margin-top:35px}</style>',
           '<h1>Rust word-OCR vertical slice</h1><p>DB ResNet34 + PARSeq, same pretrained weights, detector 1024, recognition batch 128, FP32 with TF32 disabled. Three upright synthetic pages: normal text, large text and white on black. Each backend ran in a separate process, sequentially, for at least 30 measured seconds after warmup. This is a word-only in-memory pipeline; decoding files, model loading, warmup, JSON writing, rotation and document-layout construction are excluded. It is not directly comparable with the earlier full streaming benchmark.</p>']
    parts.append(table(['Implementation','Pages/s','Measured seconds','Corpus passes','Exact words / truth'],[[name,f'{r["pages_per_second"]:.3f}',f'{sum(r["trials_seconds"]):.1f}',str(len(r['trials_seconds'])),f'{all_scores[name]["exact"]} / {all_scores[name]["truth"]}'] for name,r in results.items()]))
    ratio_ort=results['rust']['pages_per_second']/results['python_ort']['pages_per_second']
    ratio_torch=results['rust']['pages_per_second']/results['python_torch']['pages_per_second']
    parts.append(f'<p>Rust is <b>{ratio_ort:.2f}×</b> the Python/ORT throughput and <b>{ratio_torch:.2f}×</b> the Python/PyTorch throughput in this limited test. The same-ORT comparison isolates the surrounding implementations more closely; it is still not a pure GIL experiment. PARSeq export runs full-length decoding, while native PyTorch can terminate early. Rust is currently sequential between stages, with no cross-page batching or overlap.</p>')
    parts.append('<h2>Accuracy against labelled words (IoU ≥ 0.5)</h2>'+table(['Implementation','Page','Truth','Predicted','Matched boxes','Exact words'],all_rows))
    parts.append('<h2>Output parity against Python (IoU ≥ 0.99)</h2>'+table(['Reference','Page','Reference words','Rust words','Strict box matches','Identical matched text','Max coordinate delta'],[[p['reference'],p['page'],str(p['truth']),str(p['predicted']),str(p['matched']),str(p['exact']),str(p['max_matched_coordinate_delta'])] for p in parity]))
    parts.append('<p>Unmatched strict boxes and differing text are not hidden by aggregate accuracy. Full comparisons are in comparison.json. Floating-point/resize differences mean this is not a claim of universal bit-for-bit parity.</p>')
    parts.append('<h2>Control-page intermediate tensors: Rust versus Python/ORT</h2>'+table(['Tensor','Identical values','Mean absolute difference','Maximum absolute difference'],[[name,pct(n['identical_fraction']),f'{n["mean_abs"]:.7g}',f'{n["max_abs"]:.7g}'] for name,n in numerical.items()]))
    stages=results['rust']['timed_stage_totals_seconds'];count=len(results['rust']['pages'])*len(results['rust']['trials_seconds'])
    parts.append('<h2>Rust measured stage time</h2>'+table(['Stage','Milliseconds / page'],[[name,f'{seconds/count*1000:.2f}'] for name,seconds in stages.items()]))
    parts.append('<p>Recognition includes crop resizing, batching, transfers, model execution, output copying and decoding. Stage timers are wall time; allocation/drop/loop overhead can make their sum differ slightly from total time. GPU execution was separately confirmed by ORT profiling: detector nodes used CUDA; PARSeq mixed CUDA compute with CPU index/shape operations. The profile run was excluded from these throughput results.</p>')
    parts.append('<h2>Current boundaries</h2><p>Upright axis-aligned word OCR only. Crops wider than aspect ratio 8 fail explicitly pending split/remap support. No tiling, orientation recovery, retries, lines/blocks, concurrent stage queues or PDF rendering. Rectangular input is not exposed in this first port. The native executable uses CUDA/ORT DLLs from the existing environment but does not execute Python. See RUST_SLICE.md for setup, commands and the VS Code GPU-Z task.</p>')
    (root/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps({name:audit[name] for name in results},indent=2))
    print(root/'report.html')


if __name__=='__main__':main()
