"""Render the controlled resolution study as a standalone offline report."""
import json
from pathlib import Path
from pybaseline.report import table, esc, pct


def main():
    root=Path('pybaseline/results/resolution')
    study=json.loads((root/'study.json').read_text())
    rows=study['results']
    parts=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Detector resolution study</title>',
           '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#19283b;background:#f4f6fa;overflow-wrap:anywhere}p{line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #ccd}th{background:#23374e;color:white}h2{margin-top:35px}</style>',
           '<h1>Does increasing detector resolution hurt easy text?</h1>',
           '<p>Same upright A4 control page, 11-point Helvetica, '+str(study['truth_words'])+' labelled words, unchanged text and layout. Both FAST base and DB ResNet34 use their existing pretrained weights; PARSeq is fixed. No tiling, no rotation processing. Actual detector input tensors are recorded by a forward hook. Three synchronized timing trials follow one warmup; these short latency measurements are not sustained-throughput benchmarks.</p>',
           '<h2>Detector-size sweep: fixed 600-DPI source</h2><p>The high-resolution raster is held constant. Increasing the detector size increases the number of pixels per glyph. This avoids confounding the result with changing PDF rendering quality. Box matching uses IoU 0.5 against font-metric labels.</p>']
    headers=['Detector','Input H × W','Box matched','Exact words','Precision','Recall','F1','Full OCR seconds','Torch peak GiB']
    body=[]
    for r in rows:
        if r.get('source_dpi')!=600: continue
        s=r['scores']['0.5']
        body.append([esc(r['det']),f'{r["size"]} × {r["size"]}',str(s['matched']),str(s['exact']),pct(s['precision']),pct(s['recall']),pct(s['f1']),f'{r["median_seconds"]:.3f}',f'{r["allocated_peak_bytes"]/2**30:.2f}'])
    parts.append(table(headers,body))
    parts.append('<p><b>Observed result:</b> FAST retains all 900 matched boxes from 1280 through 3072, but exact recognized words fall to 885 at 3072. DB ResNet34 falls from 900 matched / 900 exact at 1024 to 894 matched / 889 exact at 3072. Larger input therefore is not monotonically better. In this fixed-source sweep the recognizer is unchanged: detector-dependent crop geometry can change recognition even when a box still passes IoU 0.5. These results do not isolate training-distribution mismatch as the cause.</p>')
    parts.append('<h2>Rendering-DPI control</h2><p>The same vector page is rerendered at 100, 200, 400 and 600 DPI, while the detector target is held at either 1024 or 2048. Higher source DPI alone does not increase the detector’s glyph scale after resizing. It can still change rasterization and interpolation quality. Recognition crops are extracted from the source raster, so exact-text changes can also involve recognition.</p>')
    body=[]
    for r in sorted((r for r in rows if r.get('size') in [1024,2048]),key=lambda r:(r['det'],r['size'],r['source_dpi'])):
        s=r['scores']['0.5']
        body.append([esc(r['det']),str(r['size']),str(r['source_dpi']),str(s['matched']),str(s['exact']),pct(s['f1'])])
    parts.append(table(['Detector','Target side','Source DPI','Box matched','Exact words','Box F1'],body))
    parts.append('<h2>Square versus rectangle: identical content pixels</h2><p>This separate detector-only test first creates the square preprocessed tensor, then removes right-hand padding to the next multiple of 32. Text pixels, their position and their scale remain identical. Both cases are mapped explicitly back to source coordinates, bypassing docTR’s square-only unpadding helper. Recognition and page preprocessing are excluded from these timings; detector postprocessing is included. Removing padding changes the boundary context, so identical predictions are not guaranteed.</p>')
    body=[]
    for r in rows:
        if r.get('mode')!='padding_ablation':continue
        s=r['score']
        body.append([esc(r['det']),f'{r["height"]} × {r["width"]}',str(s['matched']),str(s['predicted']),pct(s['f1']),f'{r["median_seconds"]:.3f}'])
    parts.append(table(['Detector','Actual H × W','Box matched','Predicted boxes','F1','Detector seconds'],body))
    parts.append('<h2>Bounding-box sensitivity</h2><p>Strict font-box matching can change when an otherwise correct word box tightens around its ink. These are the same predictions rescored, not additional inference.</p>')
    body=[]
    for r in rows:
        if r.get('source_dpi')!=600:continue
        body.append([esc(r['det']),str(r['size'])]+[str(r['scores'][str(t)]['matched']) for t in [.25,.5,.75]])
    parts.append(table(['Detector','Target side','Matched @ .25','Matched @ .5','Matched @ .75'],body))
    parts.append('<h2>Interpretation and limits</h2><p>A successful larger tensor does not mean the weights adapt to scale: the filters and receptive fields remain fixed. Glyph size, spacing, context, interpolation and postprocessing all matter. This one easy page can reveal a counterexample to “larger never hurts”; it cannot prove that any resolution is universally safe. A4 square padding wastes about 29% of tensor area; long receipts waste much more. Rectangular batches are possible when all items in a batch share dimensions, but docTR 1.1.0’s stock padding-coordinate correction assumes a square. Production rectangular inference needs explicit coordinate handling or an upstream fix.</p>')
    parts.append('<p>Sources: <a href="https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/detection/fast/pytorch.py">FAST forward implementation</a>; <a href="https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/detection/_utils/base.py">docTR coordinate unpadding</a>. Raw per-case predictions and actual tensor shapes are saved beside this report.</p>')
    (root/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    print(root/'report.html')


if __name__=='__main__': main()
