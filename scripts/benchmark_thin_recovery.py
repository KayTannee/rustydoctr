"""Native thin recovery: quality parity and warmed AB/BA throughput comparison."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tune_dense_detection import read,write,environment
from scripts.audit_missing_words import analyse
from pybaseline.metrics import polygon


def report(out):
    runs=read(out/'runs.json');pages=read(out/'manifest.json')
    base=next(r for r in reversed(runs) if not r['enabled']);on=next(r for r in reversed(runs) if r['enabled'])
    b=read(out/base['folder']/'summary.json')['first_pages'];a=read(out/on['folder']/'summary.json')['first_pages']
    quality=[]
    for i,page in enumerate(pages):
        before=b[str(i)]['words'];after=a[str(i)]['words'];kept=[w for w in after if not w.get('thin_recovery')];added=[w for w in after if w.get('thin_recovery')]
        same_text=[w['text'] for w in kept]==[w['text'] for w in before]
        same_boxes=len(kept)==len(before) and np.allclose([w['polygon'] for w in kept],[w['polygon'] for w in before],atol=1e-6,rtol=0)
        old=analyse(page,before);new=analyse(page,after);correct=0;details=[]
        for word in added:
            shape=polygon(word.get('quadrilateral',word['polygon']))
            hits=[t for t in page['words'] if polygon(t['polygon']).area and shape.intersection(polygon(t['polygon'])).area/polygon(t['polygon']).area>=.5]
            good=len(hits)==1 and hits[0]['text']==word['text'];correct+=good
            details.append(dict(word=word,truth=[t['text'] for t in hits],correct=good))
        count=lambda rows:sum(w['group']=='ordinary' and w['status']=='no_box_overlap' for w in rows)
        quality.append(dict(page=page['id'],existing_text_unchanged=same_text,existing_boxes_unchanged=bool(same_boxes),added=len(added),correct_added=correct,other_added=len(added)-correct,missing_before=count(old),missing_after=count(new),details=details))
    write(out/'quality.json',quality)
    def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Native thin recovery</title><style>body{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#203247}p{line-height:1.5}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd}th{background:#e0edf5}.scroll{overflow:auto}</style><h1>Native thin recovery</h1><p>Same resident DB ResNet34 + PARSeq models, detector 1536, dense tiles, CPU orientation/deskew, recognition batch 256, two postprocess workers and three admitted pages. Recovery reuses detector masks and the recognition queue. Loading and a full workload warmup are excluded from measured seconds; decode, inference, ordered output and drain are included. Device peak is sampled during measurement; incremental peak covers loading/warmup too and is relative to each run’s pre-load idle.</p>']
    parts.append(table(['Run','Recovery','Pages','Seconds','Pages/s','Measured device peak GiB','Whole-run incremental peak GiB','Mean GPU %','Max admitted'],[[r['folder'],r['enabled'],r['pages'],round(r['seconds'],2),round(r['pps'],3),round(r['resources']['device_vram_bytes_max']/2**30,2),round(r['resources']['vram_increment_peak_bytes']/2**30,2),round(r['resources']['gpu_util_pct_mean'],1),r['max_inflight']] for r in runs]))
    timed=[r for r in runs if r['pages']>len(pages)]
    if timed:
        rates={mode:sum(r['pages'] for r in timed if r['enabled']==mode)/sum(r['seconds'] for r in timed if r['enabled']==mode) for mode in [False,True]}
        parts.append(f'<p>Pooled sustained throughput: off {rates[False]:.3f}, on {rates[True]:.3f} pages/s; {(rates[True]/rates[False]-1)*100:+.2f}%. Two orders do not establish statistical significance for small differences.</p>')
        write(out/'throughput.json',dict(off_pps=rates[False],on_pps=rates[True],change_pct=(rates[True]/rates[False]-1)*100))
    parts.append('<h2>Quality</h2><p>17 synthetic variants of nine layouts; repeated content at different angles. Correct addition means exact text and at least half of one labelled word covered; it is not the strict IoU metric. Native full-page recovery also runs on older pages whose prototype caches contained only tiles.</p>')
    parts.append(table(['Page','Added','Correct','Other','Ordinary omissions before','After','Existing text/boxes unchanged'],[[r['page'],r['added'],r['correct_added'],r['other_added'],r['missing_before'],r['missing_after'],r['existing_text_unchanged'] and r['existing_boxes_unchanged']] for r in quality]))
    parts.append('<p><a href="quality.json">Word-level evidence</a> &middot; <a href="runs.json">Timing/resources</a> &middot; <a href="provenance.json">Input hashes</a></p>')
    if (out/'repeat_checks.json').exists():
        checks=read(out/'repeat_checks.json')
        parts.append('<h2>Repeated-output checks</h2><p>Ordered drain and text/coordinate consistency against each first result. Coordinates compare at native f32 precision; confidence values need not be bit-identical.</p>')
        parts.append(table(['Run','Pages','Changed pages','Recovered words'],[[r['run'],r['pages'],r['changed_pages'],r['recovered_words']] for r in checks]))
        parts.append('<p><a href="repeat_checks.json">Repeat-check evidence</a></p>')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8')
    print('Quality:',dict(Counter({k:sum(r[k] for r in quality) for k in ['added','correct_added','other_added','missing_before','missing_after']})),flush=True)
    assert all(r['existing_text_unchanged'] and r['existing_boxes_unchanged'] for r in quality),'Existing words changed; inspect quality.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pages',type=int,default=340)
    parser.add_argument('--quality-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output or ROOT/'pybaseline/results'/('native_thin_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.report_only:report(out);return
    if args.resume:assert (out/'runs.json').exists()
    else:
        out.mkdir(parents=True,exist_ok=False)
        pages=read(ROOT/'pybaseline/results/dense_tuning_v2/development/manifest.json')+read(ROOT/'pybaseline/results/dense_tuning_v2/validation/manifest.json')+read(ROOT/'pybaseline/results/thin_fresh_v1/baseline/manifest.json')
        write(out/'manifest.json',pages);write(out/'workload.json',dict(pages=[dict(id=p['id'],image=p['image']) for p in pages]));write(out/'runs.json',[])
        files=[ROOT/'target/release/throughput.exe',*[ROOT/'models'/n for n in ['db_resnet34.onnx','parseq.onnx','metadata.json','page_orientation.onnx','page_orientation.json']],*[Path(p['image']) for p in pages]]
        write(out/'provenance.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    pages=read(out/'manifest.json');runs=read(out/'runs.json')
    assert args.pages>=len(pages) and args.pages%len(pages)==0
    plans=[('quality_off',False,len(pages)),('quality_on',True,len(pages))]
    if not args.quality_only:plans += [('1_off',False,args.pages),('1_on',True,args.pages),('2_on',True,args.pages),('2_off',False,args.pages)]
    for name,enabled,count in plans:
        if any(r['folder']==name for r in runs):continue
        cmd=[str(ROOT/'target/release/throughput.exe'),'--workload',str((out/'workload.json').resolve()),'--output',str((out/name).resolve()),'--pages',str(count),'--size','1536','--reco-batch','256','--det-batch','1','--workers','2','--inflight','3','--arena-mib','6144','--det-arena-mib','4096','--page-orientation','--deskew','--dense-refine']
        if enabled:cmd.append('--thin-recovery')
        binary_hash=hashlib.sha256((ROOT/'target/release/throughput.exe').read_bytes()).hexdigest()
        print('Run',name,flush=True);subprocess.run(cmd,cwd=ROOT,env=environment(),check=True,timeout=1800)
        s=read(out/name/'summary.json');assert s['pages']==count and s['max_inflight']<=3
        runs.append(dict(folder=name,enabled=enabled,pages=count,seconds=s['wall_seconds'],pps=s['pages_per_second'],resources=s['resources'],max_inflight=s['max_inflight'],stage_seconds=s['stage_seconds'],binary_sha256=binary_hash))
        write(out/'runs.json',runs)
        if name=='quality_on':report(out)
    report(out);print(out/'report.html')


if __name__=='__main__':main()
