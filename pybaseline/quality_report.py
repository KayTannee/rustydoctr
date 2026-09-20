"""Build an accuracy-only report. Shared-GPU runs are never throughput benchmarks."""
import html,json
from pathlib import Path
from pybaseline.metrics import score_words, polygon

OUT=Path('pybaseline/results/quality');ROOT=Path('output/pdf/quality')
def load(name):return json.loads((OUT/name).read_text())

def merge_evidence(page, words):
 evidence=[]
 truth=[polygon(w['polygon']) for w in page['words']]
 for word in words:
  box=polygon(word['polygon'])
  covered=[w['text'] for w,g in zip(page['words'],truth) if g.area and box.intersection(g).area/g.area>=.5]
  if len(covered)>1:evidence.append(dict(truth=covered,recognized=word['text'],confidence=word['confidence']))
 return sorted(evidence,key=lambda r:r['confidence'],reverse=True)
def table(headers,rows):
 def cells(row,tag):return ''.join(f'<{tag}>{html.escape(str(c))}</{tag}>' for c in row)
 return '<div class="scroll"><table><thead><tr>'+cells(headers,'th')+'</tr></thead><tbody>'+''.join('<tr>'+cells(r,'td')+'</tr>' for r in rows)+'</tbody></table></div>'
def overlay(page,words):
 w,h=page['width'],page['height'];items=[f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"><image href="../../../output/pdf/quality/{page["image"]}" width="{w}" height="{h}"/>']
 for group,color in [(page['words'],'#09904a'),(words,'#e03e2d')]:
  for word in group:
   p=word['polygon']
   if len(p)==2:
    (x0,y0),(x1,y1)=p;p=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
   points=' '.join(f'{x*w:.2f},{y*h:.2f}' for x,y in p)
   items.append(f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="3"><title>{html.escape(word["text"])}</title></polygon>')
 items.append('</svg>');return ''.join(items)
def main():
 manifest=json.loads((ROOT/'manifest.json').read_text());pages={p['id']:p for p in manifest['pages']};scale=load('scale_results.json');rotation=load('rotation_results.json');refine=load('refinement_results.json');cpu=load('orientation_cpu.json');chars=load('isolated_characters.json');local=load('local_results.json') if (OUT/'local_results.json').exists() else []
 parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>OCR quality experiments</title>
 <style>body{font:16px system-ui;max-width:1180px;margin:35px auto;padding:0 22px;background:#f5f7fb;color:#203247}p,li{line-height:1.6}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px 12px;border-bottom:1px solid #d6dde6;text-align:left}th{background:#223d58;color:white}.scroll{overflow:auto}h2{margin-top:36px}.notice{padding:15px;background:#e5eef9;border-left:4px solid #3478bd}svg{display:block;width:100%;max-width:750px;margin:auto}summary{cursor:pointer;padding:12px;background:#e3eaf3;margin-top:8px}code{background:#e3eaf3;padding:2px 5px}</style>
 <h1>Word segmentation and rotation: accuracy experiments</h1>
 <p class="notice">Accuracy only. Another application was using the GPU. No throughput, latency or utilization conclusions are drawn from these runs. The production Rust pipeline is unchanged; all new policies are experimental Python/ONNX prototypes.</p>
 <p>DB ResNet34 + PARSeq, identical FP32 ONNX weights to the native baseline. Three synthetic A4 statements and a text-size ladder, rendered at 300 DPI. Main scores require one-to-one word matching at IoU ≥ 0.5 and exact text; diagnostic counts at IoU ≥ 0.25 are identified separately. Labels use font metrics, so tight boxes around I or punctuation can fail the strict overlap test.</p>''']
 rows=[]
 for r in scale:
  if r['page']=='text_scale':continue
  s=r['score'];rows.append([r['page'],r['size'],s['truth'],s['matched'],s['exact'],r['by_region']['legal']['exact'],s['merge_candidates'],s['split_candidates']])
 parts.append('<h2>Full-page resolution</h2>'+table(['Page','Detector size','Truth','Matched','Exact','Legal exact','Merge candidates','Split candidates'],rows))
 parts.append('<p>Increasing detector resolution strongly helps the dense legal text. It does not fix locally rotated text in an upright-only crop pipeline, and gains are not uniform across regions. This changes the tensor given to the detector; increasing source DPI alone is a different experiment.</p>')
 ladder=pages['text_scale'];rows=[]
 for sizept in [144,96,72,48,36,24,18,12,10,8,6]:
  row=[sizept]
  for size in [1024,1536,2048,2560]:
   r=next(r for r in scale if r['page']=='text_scale' and r['size']==size);pairs=dict(score_words(ladder['words'],r['words'],.25,True)['pairs'])
   index=next(i for i,w in enumerate(ladder['words']) if w['region']==f'pt_{sizept}' and w['text']=='Sit')
   row.append(r['words'][pairs[index]]['text'] if index in pairs else 'no complete match')
  rows.append(row)
 parts.append('<h2>Does bigger detector input ever hurt?</h2>'+table(['Font pt / truth “Sit”','1024','1536','2048','2560'],rows))
 parts.append('<p>Yes, in this controlled ladder. The 96 pt word is correct at 1536, then loses its complete match at 2048/2560; at 72 pt it loses its complete match at 2560. The largest glyphs break into fragments or partial detections. This establishes scale sensitivity for these examples, not a universal maximum font size. The repeated phrase also exposes I/a merges at small and medium sizes.</p>')
 for r in [r for r in scale if r['page']=='text_scale']:
  parts.append(f'<details><summary>Inspect ladder at {r["size"]}: green truth / red predictions (hover boxes for text)</summary>'+overlay(ladder,r['words'])+'</details>')
 rows=[]
 for r in refine:
  baseline=next(s for s in scale if s['page']==r['page'] and s['size']==1536);high=next(s for s in scale if s['page']==r['page'] and s['size']==2560)
  rows.append([r['page'],baseline['score']['exact'],r['scores']['0.5']['exact'],high['score']['exact'],len(r['policy']['regions']),r['policy']['detector_pixels']])
 parts.append('<h2>Selective dense-region rescanning</h2>'+table(['Page','Base 1536 exact','Refined exact','Full 2560 exact','Extra tiles','Detector input pixels'],rows))
 parts.append('<p>The heuristic uses small connected ink components and their density; it is not given legal-region labels, font sizes or ground-truth coordinates. It selects one band and rescans two overlapping half-width tiles at 1024. Existing words outside the band are retained; seam ownership prevents double insertion. It selects no region on the size ladder, preserving the base output there.</p><p>For each statement this evaluates 4.46 million detector-input pixels versus 6.55 million at full-page 2560, about 32% fewer. That is a work-count proxy, <b>not a measured speedup</b>; the prototype also redundantly recognizes crops and is not the final scheduler. A 2048 full-page pass uses 4.19 million pixels, so it remains an important comparator. Region replacement can lose individual words and needs further validation.</p>')
 evidence=merge_evidence(pages['statement_2_cw0'],next(r['words'] for r in scale if r['page']=='statement_2_cw0' and r['size']==1536))
 (OUT/'merge_evidence.json').write_text(json.dumps(evidence,indent=2))
 parts.append('<h2>Why recognition confidence is insufficient</h2>'+table(['Ground-truth words covered by a box','Recognized','Confidence'],[[' | '.join(x['truth']),x['recognized'],f"{x['confidence']:.5%}"] for x in evidence[:9]]))
 parts.append('<p>These are overlapping-ground-truth diagnostics, not an inference-time oracle. They show confident dropped or merged words. A real trigger must use image gaps, component density, uncovered text-like ink and detector geometry, not just recognition confidence.</p>')
 errors=[abs(r['raw_skew']-((r['truth_cw']+45)%90-45)) for r in cpu]
 correct=sum((r['classifier_angle']-round(r['truth_cw']/90)*90)%360==0 for r in cpu)
 parts.append(f'<h2>Page direction and fractional skew</h2><p>The CPU page classifier identified the coarse orientation in {correct}/{len(cpu)} cases. A weighted line-angle estimate retained fractional values; mean absolute residual error was {sum(errors)/len(errors):.3f}°, maximum {max(errors):.3f}°. These ruled statement pages make skew estimation easier than sparse or unruled scans.</p>')
 parts.append(table(['Actual clockwise angle','Estimated residual','Uncorrected exact','Rounded correction exact','Fractional correction exact'],[[r['truth_cw'],f"{r['fine_skew']:.3f}°",*[r['variants'][v]['scores']['0.5']['exact'] for v in ['uncorrected','rounded','fractional']]] for r in rotation]))
 parts.append('<p>The rounded comparison rounds our estimated residual; it is an ablation of quantization, not a claim to reproduce the entire docTR orientation pipeline. The installed docTR estimator explicitly rounds its angle. Fractional correction helps the ±0.5° examples, but slightly hurts some ±1.5° examples: interpolation, residual estimation and detector sensitivity still matter. Coarse orientation plus skew uses one final detector pass, rather than a full OCR pass followed by another full-page OCR pass.</p>')
 if local:
  rows=[]
  for r in local:
   for name,v in r['modes'].items():rows.append([r['page'],name,v['scores']['0.5']['exact'],v['regions_iou25']['local90']['exact'],v['regions_iou25']['local45']['exact'],sum(s['truth']==s['predicted'] for s in v['single_characters']),len(v['single_characters'])])
  parts.append('<h2>Local orientation: reuse one detector map</h2>'+table(['Page','Crop policy','Exact IoU .5','90° exact / 7','45° exact / 3','Single chars exact .25','Single chars truth'],rows))
  parts.append('<p>Both policies postprocess the same detector probability map as quadrilaterals. “Independent” applies the crop classifier to each docTR-style rectified crop. “Line prior” uses elongated word anchors and nearby line alignment; short ambiguous crops inherit that direction, falling back to upright without an anchor. Anchor classifiers can resolve a 180° flip. This is a prototype, not a complete docTR-versus-Rust comparison; crop geometry and orientation policy both change.</p>')
 if (OUT/'rotated_roi_results.json').exists():
  rescans=load('rotated_roi_results.json')
  parts.append('<p class="notice"><b>Rejected as a default:</b> on the densest statement, a false diagonal candidate in upright legal text was accepted and replaced good words. Exact words fell from 404 to 392. Recognized text length and confidence are insufficient acceptance gates. Preserve existing results until direction evidence and replacement checks justify changing them.</p>')
  parts.append('<h2>Recovering text the original detector missed</h2>'+table(['Page','Selected image regions','Extra detector size','Exact IoU .5','90 degree exact / 7','45 degree exact / 3'],[[r['page'],len(r['regions']),r.get('roi_size',256),r['scores']['0.5']['exact'],r['regions_iou25']['local90']['exact'],r['regions_iou25']['local45']['exact']] for r in rescans]))
  parts.append('<p>Local orientation policies alone recovered none of the complete rotated words. A separate image-component pass looks for uncovered non-horizontal groups, rejects isolated thin bars and uses no labelled regions. It straightens and detects only the selected regions, retaining other page results. This makes detection recovery possible without rerunning the whole page.</p><p>An initial 256-input trial recovered three sideways reference words but clipped part of the diagonal label. The revised trial includes intersecting existing word boxes in the crop extent and uses 512 input. Both geometry and resolution changed; this is exploratory debugging, not an isolated resolution ablation. Initial results are retained in rotated_roi_256_results.json. These are development fixtures, with limits on candidate count, and the full-page direction must already be corrected.</p>')
 parts.append('<h2>Isolated-character diagnostic</h2><p>Using ground-truth upright crops isolates recognition from detection. This is an oracle diagnostic, not an implementable detection policy.</p>'+table(['Truth','Region','Upright recognition','Crop-classifier angle','Classifier confidence'],[[r['truth'],r['region'],r['upright_recognized'],r['crop_orientation'],r['orientation_confidence']] for r in chars]))
 parts.append('<p>Nonzero classifications of symmetric I/dash crops are not automatically harmful errors. A tightly cropped underscore can already be read as a dash without rotation. Preserve line direction and baseline position, and inspect crop geometry before blaming the orientation classifier alone.</p>')
 parts.append('<h2>Next implementation decision</h2><p>Keep the existing upright fast path. Prototype page-level direction and float skew before detection; retain detector maps for oriented geometry; select bounded refinement regions and reconcile detections before recognizing final crops. Keep both models resident and reuse the existing queues. Do not enable these heuristics by default yet: all three statement layouts have now been inspected and are development examples, not an untouched hold-out. Add unseen layouts, unruled/sparse scans and opposite local text directions before selecting defaults.</p><p>Raw predictions and diagnostics are alongside this report. Timing fields are diagnostic only and deliberately omitted here.</p></html>')
 (OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
 summary=dict(scale=[{k:r[k] for k in ['page','size','score','by_region']} for r in scale],refinement=[{k:r[k] for k in ['page','policy','scores']} for r in refine],page_orientation_correct=correct,page_orientation_total=len(cpu),skew_mae_deg=sum(errors)/len(errors),skew_max_error_deg=max(errors),performance_measured=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(OUT/'report.html')
if __name__=='__main__':main()
