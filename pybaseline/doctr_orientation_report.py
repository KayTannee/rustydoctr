"""Report native docTR rotation versus Rust; accuracy and pass counts, not speed."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
from pybaseline.quality_report import table,overlay

NAMES={
 'doctr_upright':'docTR: upright control',
 'doctr_crops':'docTR: crop rotation',
 'doctr_straight_upright':'docTR: preserve-coords straighten + upright crops',
 'doctr_straight_crops':'docTR: preserve-coords straighten + crop rotation',
 'doctr_default_upright':'docTR: default straighten + upright crops',
 'doctr_default_crops':'docTR: default straighten + crop rotation',
 'doctr_default_no_crop':'docTR: same oriented boxes, crop classifier disabled',
 'rust_base':'Rust: upright control',
 'rust_quarter':'Rust: coarse page direction',
 'rust_deskew':'Rust: coarse + fractional deskew',
 'rust_refined':'Rust: coarse + deskew + dense refinement',
}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory.resolve()
 rows=[]
 for mode in NAMES:
  if mode.startswith('doctr_'):rows.extend(json.loads((out/f'{mode}.json').read_text()))
 rows.extend(json.loads((out/'rust.json').read_text()))
 statements=[r for r in rows if r['page']!='text_scale']
 parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>docTR versus Rust rotation</title><style>body{font:16px system-ui;color:#203247;background:#f5f7fb;max-width:1350px;margin:30px auto;padding:0 20px}p,li{line-height:1.6}th,td{padding:9px 12px;border-bottom:1px solid #d6dde6;text-align:left}th{background:#223d58;color:white}table{border-collapse:collapse;background:white;width:100%}.scroll{overflow:auto}h2{margin-top:32px}.note{background:#e5eef9;padding:14px;border-left:4px solid #3478bd}summary{padding:12px;background:#e3eaf3;cursor:pointer}svg{width:100%;max-width:1000px}</style>
 <h1>Python docTR versus Rust: rotation and deskew</h1>
 <p class="note">Actual installed docTR 1.1.0 OCRPredictor on CUDA, DB ResNet34 + PARSeq FP32, detector 1536, recognition batch 256, TF32 disabled. Three synthetic statements at nine rotations each (12,006 labelled words), plus a separate size ladder. These are development fixtures, not independent real-document validation.</p>
 <p>docTR uses its native PyTorch decoder and classifiers; Rust uses exported ONNX detector/recognizer graphs and a CPU page classifier. Rust predictions are reused from the prior native run after checking model and manifest hashes. No throughput ratio is inferred from these accuracy runs. docTR's observed full-page detector pass counts are reported directly.</p>''']
 summary=[]
 for mode,name in NAMES.items():
  rr=[r for r in statements if r['mode']==mode];exact=sum(r['scores']['0.5']['exact'] for r in rr);matched=sum(r['scores']['0.5']['matched'] for r in rr)
  singles=[s for r in rr for s in r['single_characters']];single_exact=sum(s['truth']==s['predicted'] for s in singles)
  local90=sum(r['regions']['local90']['exact'] for r in rr);local45=sum(r['regions']['local45']['exact'] for r in rr)
  passes=sorted({r['trace']['detector_passes'] for r in rr}) if mode.startswith('doctr') else ('1 + bounded tiles' if mode=='rust_refined' else '1')
  summary.append(dict(mode=mode,exact=exact,matched=matched,single_exact=single_exact,single_truth=len(singles),local90=local90,local45=local45,detector_passes=passes))
 parts.append('<h2>Overall accuracy</h2>'+table(['Mode','Exact / 12006','Exact recall','Matched IoU .5','Single chars exact .25','90° local exact / 189','45° local exact / 81','Detector passes/page'],[[NAMES[r['mode']],r['exact'],f"{r['exact']/12006:.2%}",r['matched'],f"{r['single_exact']}/{r['single_truth']}",r['local90'],r['local45'],r['detector_passes']] for r in summary]))
 parts.append('<p>Main scores require one-to-one word matching at IoU ≥ 0.5 and exact text. Single-character and local-rotation diagnostics use IoU ≥ 0.25 because labels use font metrics rather than tight ink. The refined Rust row adds a segmentation policy absent from the other modes; compare unrefined Rust rows when assessing rotation alone.</p>')
 by_mode={r['mode']:r for r in summary};best=max((r for r in summary if r['mode'].startswith('doctr')),key=lambda r:r['exact']);difference=by_mode['rust_deskew']['exact']-best['exact']
 parts.append(f'<p class="note">Rust coarse + deskew versus the best tested docTR configuration: {difference:+d} exact words ({difference/12006*100:+.2f} percentage points). Both use the same pretrained detector/recognizer pair. Rust does not demonstrate a material rotation-accuracy advantage here. Its architectural advantage is one full-page detector pass; this is not a measured speedup claim.</p>')
 groups={'Upright':lambda a:a==0,'±0.5°':lambda a:abs(a)==.5,'±1.5°':lambda a:abs(a)==1.5,'90/180/270°':lambda a:a in [90,180,270],'90.5°':lambda a:a==90.5}
 parts.append('<h2>Exact words by page rotation</h2>'+table(['Mode',*groups],[[name,*[sum(r['scores']['0.5']['exact'] for r in statements if r['mode']==mode and predicate(r['truth_angle'])) for predicate in groups.values()]] for mode,name in NAMES.items()]))
 parts.append('<h2>Reported page angles</h2><p>docTR detect_orientation reports its combined coarse/skew estimate; it does not itself straighten the page. straighten_pages enables the second detection pass. Rust records both estimated and applied fractional skew. All angles below use clockwise input orientation, normalized to [-180, 180).</p>')
 angle_rows=[]
 for page in dict.fromkeys(r['page'] for r in statements):
  rr={r['mode']:r for r in statements if r['page']==page};d=rr['rust_deskew']['geometry']
  values=[rr[m]['page_orientation']['value'] for m in ['doctr_crops','doctr_straight_upright','doctr_straight_crops']]
  angle_rows.append([page,rr['rust_deskew']['truth_angle'],*values,(d['applied_quarter_deg']+d['skew']['estimated_deg']+180)%360-180,(d['applied_quarter_deg']+d['skew']['applied_deg']+180)%360-180])
 parts.append(table(['Page','Truth','docTR crops','docTR straighten/upright','docTR straighten/crops','Rust estimate','Rust applied'],angle_rows))
 parts.append('<h2>Single-character confusions</h2><p>Counts include every occurrence across all 27 variants, so rotated copies are not independent examples. Unmatched means no one-to-one box match, which can include missing or merged characters. A recognizer error here is not automatically caused by the crop classifier.</p>')
 confusion=[]
 for mode,name in NAMES.items():
  singles=[s for r in statements if r['mode']==mode for s in r['single_characters']]
  for char in ['I','a','-','_','—']:
   counts=Counter(s['predicted'] for s in singles if s['truth']==char)
   confusion.append([name,char,sum(counts.values()),counts[char],counts['<unmatched>'],'; '.join(f'{text}: {n}' for text,n in counts.most_common() if text not in [char,'<unmatched>'])])
 parts.append(table(['Mode','Truth','Count','Correct','Unmatched','Other predictions'],confusion))
 c=[s for r in statements if r['mode']=='doctr_default_crops' for s in r['single_characters']];d=[s for r in statements if r['mode']=='doctr_default_no_crop' for s in r['single_characters']]
 parts.append(f'<p>The crop-classifier ablation keeps the same oriented-box geometry and page straightening. Disabling the classifier changes total exact words by {by_mode["doctr_default_no_crop"]["exact"]-by_mode["doctr_default_crops"]["exact"]:+d}. Matched I→dash cases are {sum(s["truth"]=="I" and s["predicted"]=="-" for s in c)} with it enabled and {sum(s["truth"]=="I" and s["predicted"]=="-" for s in d)} disabled. Geometry/recognition errors remain without the classifier; disabling it alone is insufficient.</p>')
 parts.append('<h2>Every page</h2>'+table(['Page',*[NAMES[m] for m in NAMES]],[[page,*[next(r['scores']['0.5']['exact'] for r in rows if r['mode']==mode and r['page']==page) for mode in NAMES]] for page in dict.fromkeys(r['page'] for r in rows)]))
 geometry_rows=[]
 for mode in NAMES:
  if not mode.startswith('doctr'):continue
  rr=[r for r in statements if r['mode']==mode]
  geometry_rows.append([NAMES[mode],sum(r['scores']['0.5']['exact'] for r in rr),sum(r['returned_geometry_score']['exact'] for r in rr) if rr[0]['returned_geometry_score'] is not None else 'rectified coordinates: not directly comparable'])
 parts.append('<h2>Coordinate audit</h2><p>Both preserve_original_coords settings are tested: the installed version uses different straightening implementations for them, which changes the raster itself. With preservation enabled, a read-only observer retains quadrilaterals using docTR’s own inverse before its upright boxes are reduced to envelopes. With the default setting, an observer reconstructs the pad/rotate/crop transform and asserts byte-for-byte equality with the actual rectified raster before mapping predictions back. Neither observer changes model inputs, angles or predictions. Main scores use original-image quadrilaterals.</p>'+table(['docTR mode','Quadrilateral exact','Public geometry exact'],geometry_rows))
 if (out/'raster_audit.json').exists():
  raster=json.loads((out/'raster_audit.json').read_text())
  parts.append('<p>The CPU raster audit confirms this is more than an output-coordinate difference. Extra black padding changes the effective text scale at the fixed detector input. Example quarter-turn rasters (height × width):</p>'+table(['Page','preserve_original_coords=True','Default False','Preserve black pixels','Default black pixels'],[[r['page'],r['preserve_shape'],r['default_shape'],f"{r['preserve_black_fraction']:.1%}",f"{r['default_black_fraction']:.1%}"] for r in raster if r['page'] in ['statement_1_cw0','statement_1_cw90','statement_1_cw180','statement_1_cw270']]))
 fixtures=Path('output/pdf/quality').resolve();manifest=json.loads((fixtures/'manifest.json').read_text())
 parts.append('<h2>Inspect original-coordinate predictions</h2><p>Green: labels. Red: predictions. Hover each polygon to inspect its text.</p>')
 for page_id in ['statement_0_cw0','statement_0_cw90.5']:
  page=next(p for p in manifest['pages'] if p['id']==page_id)
  for mode in ['doctr_crops','doctr_straight_upright','doctr_default_upright','doctr_default_crops','rust_deskew']:
   row=next(r for r in rows if r['mode']==mode and r['page']==page_id)
   diagram=overlay(page,row['words']).replace('../../../output/pdf/quality/',os.path.relpath(fixtures,out).replace(os.sep,'/')+'/')
   parts.append(f'<details><summary>{html.escape(page_id+" — "+NAMES[mode])}</summary>'+diagram+'</details>')
 parts.append('<h2>Limits</h2><p>This compares complete implementations on the installed library version, not the language or GIL in isolation. Model backends, crop geometry and orientation policies differ. Source hashes, options, predictions, angles and detector-call traces are retained beside this report. Single-pass diagnostic times include shape/cache variation and instrumentation; they are deliberately omitted from performance conclusions. More real scans and a matched sustained benchmark are needed before claiming general superiority.</p></html>')
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 (out/'report.html').write_text('\n'.join(parts),encoding='utf-8')
 print(out/'report.html')


if __name__=='__main__':main()
