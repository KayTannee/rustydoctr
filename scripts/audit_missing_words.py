"""CPU-only audit of saved OCR: prioritise omitted ordinary words, not stress-token splits."""
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pybaseline.metrics import polygon, score_words
from shapely.strtree import STRtree


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def priorities(words):
    """Explicit fixture annotations/sequences only; never infer exclusions from OCR errors."""
    excluded = set()
    # Preserve the ordinary sentence preceding this artificial character sequence.
    sequences = ['I a A i l 1 O 0'.split(), 'A - B; (I), [a], 1:0.'.split()]
    for seq in sequences:
        for i in range(len(words)-len(seq)+1):
            if [w['text'] for w in words[i:i+len(seq)]] == seq:
                excluded.update(range(i, i+len(seq)))
                if seq[0:2] == ['I', 'a']:
                    j = i+len(seq)
                    while j < len(words) and words[j].get('region') == 'characters':
                        excluded.add(j)
                        j += 1
    return ['stress' if i in excluded else
            'local_rotation' if w.get('region') in ('local90', 'local45') else
            'punctuation' if not any(c.isalnum() for c in w['text']) else 'ordinary'
            for i, w in enumerate(words)]


def analyse(page, predictions):
    truth = page['words']
    gt = [polygon(w['polygon']) for w in truth]
    shapes = [polygon(w.get('quadrilateral', w.get('polygon'))) for w in predictions]
    pred = [dict(w, polygon=list(s.exterior.coords)[:-1]) for w, s in zip(predictions, shapes)]
    strict = dict(score_words(truth, pred, .5, True)['pairs'])
    relaxed = dict(score_words(truth, pred, .25, True)['pairs'])
    tree = STRtree(shapes)
    covered = defaultdict(list)
    overlaps = []
    for i, g in enumerate(gt):
        hits = [(int(j), g.intersection(shapes[j]).area/g.area) for j in tree.query(g)] if g.area else []
        hits = [(j, a) for j, a in hits if a > 1e-9]
        overlaps.append(hits)
        for j, a in hits:
            if a >= .5:
                covered[j].append(i)
    rows = []
    for i, (w, group) in enumerate(zip(truth, priorities(truth))):
        j = strict.get(i, relaxed.get(i))
        if i in strict:
            status = 'exact' if pred[j]['text'] == w['text'] else 'recognition_error'
        elif i in relaxed:
            status = 'loose_exact' if pred[j]['text'] == w['text'] else 'loose_recognition_error'
        elif any(a >= .5 and len(covered[k]) > 1 for k, a in overlaps[i]):
            status = 'merged_coverage'
        elif overlaps[i]:
            status = 'partial_or_unmatched'
        else:
            status = 'no_box_overlap'
        rows.append(dict(index=i, text=w['text'], region=w.get('region'), group=group,
                         status=status, polygon=w['polygon'],
                         best_coverage=max((a for _, a in overlaps[i]), default=0),
                         nearby=[dict(index=k, text=pred[k]['text'], coverage=a,
                                      polygon=pred[k]['polygon']) for k, a in overlaps[i]]))
    return rows


def main():
    out = ROOT/'output/diagnostics/missing-words'
    out.mkdir(parents=True, exist_ok=True)
    old = ROOT/'pybaseline/results/line_orientation_v2'
    tuned = ROOT/'pybaseline/results/dense_tuning_v2'
    pages = [p for p in read(old/'manifest.json') if p['id'].startswith('statement')]
    records = {r['id']:r['words'] for r in read(old/'guided/summary.json')['first_pages'].values()}
    pages += read(tuned/'validation/manifest.json')
    records.update({r['page']:r['words'] for r in read(tuned/'validation/recognition.json') if r['setting'] == 0})
    result = []
    for p in pages:
        result.append(dict(id=p['id'], image=p['image'], width=p['width'], height=p['height'],
                           angle=p['page_rotation_deg'], rows=analyse(p, records[p['id']])))
    stats = []
    for family in ['statement', 'dense_validation']:
        for group in ['ordinary', 'stress', 'local_rotation', 'punctuation']:
            rows = [w for p in result if p['id'].startswith(family) for w in p['rows'] if w['group']==group]
            stats.append(dict(family=family, group=group, truth=len(rows), statuses=dict(Counter(w['status'] for w in rows))))
    singles = []
    for token in ['I', 'a', 'A', *map(str, range(10))]:
        rows = [w for p in result for w in p['rows'] if w['group']=='ordinary' and w['text']==token]
        singles.append(dict(token=token, truth=len(rows), statuses=dict(Counter(w['status'] for w in rows))))
    bundle = dict(statistics=stats, singles=singles, pages=result)
    # Inspect existing maps only: truth locates diagnostic windows, never production candidates.
    import cv2
    import numpy as np
    traces = []
    indexed = {p['id']:p for p in result}
    for split in ['development', 'validation']:
        cache = tuned/split/'cache'
        for info in read(cache/'cache.json'):
            page = indexed[info['id']]
            matrix = np.vstack([info['geometry']['corrected_to_original'], [0,0,1]])
            inverse = np.linalg.inv(matrix)
            for tile, filename in info['tiles']:
                prob = np.fromfile(cache/filename, dtype='<f4').reshape(1024,1024)
                mask = (prob >= .3).astype(np.uint8)
                opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
                tw,th = tile['width'],tile['height']
                rw,rh = (1024,int(1024*th/tw)) if tw>=th else (int(1024*tw/th),1024)
                offset = np.array([(1024-rw+1)//2,(1024-rh+1)//2])
                for word in page['rows']:
                    if word['group']!='ordinary' or word['status']!='no_box_overlap':
                        continue
                    pts = np.array(word['polygon'])*[page['width'],page['height']]
                    pts = (np.c_[pts,np.ones(len(pts))]@inverse.T)[:,:2]
                    cx,cy = pts.mean(axis=0)
                    if not (tile['owner_start']<=cx<tile['owner_end'] and tile['y']<=cy<tile['y']+th):
                        continue
                    pts = (pts-[tile['x'],tile['y']])*[rw/tw,rh/th]+offset
                    region = np.zeros_like(mask)
                    cv2.fillConvexPoly(region,np.round(pts).astype(np.int32),1)
                    region = region.astype(bool)
                    before,after = int(mask[region].sum()),int(opened[region].sum())
                    traces.append(dict(page=page['id'], index=word['index'], text=word['text'],
                                       peak=float(prob[region].max()),threshold_pixels=before,opened_pixels=after,
                                       stage='below_threshold' if before==0 else 'removed_by_opening' if after==0 else 'survives_opening'))
    bundle['tile_map_trace'] = traces
    (out/'audit.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    statuses = ['exact', 'recognition_error', 'loose_exact', 'loose_recognition_error', 'merged_coverage', 'partial_or_unmatched', 'no_box_overlap']
    def table(rows, keys):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys+statuses)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+''.join('<td>'+str(r['statuses'].get(k,0))+'</td>' for k in statuses)+'</tr>' for r in rows)+'</table></div>'
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Missing ordinary words</title><style>body{font:15px system-ui;color:#203247;max-width:1400px;margin:30px auto;padding:0 20px}p{line-height:1.5}.scroll{overflow:auto}table{border-collapse:collapse}th,td{padding:9px;border:1px solid #ddd}th{background:#e0edf5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}.card{border:1px solid #ccd;padding:12px}svg{width:100%;height:150px}</style><h1>Missing ordinary words</h1>',
             '<p>Priority: omissions in prose and fields, especially standalone I, a and digits. Explicit artificial character sequences and locally rotated reference strings are reported separately. Normal sentences in the character-test region remain included. No inference or correction was performed for this audit.</p>',
             '<p><b>No box overlap</b> means no predicted polygon intersects the labelled word. This is stronger evidence of omission than a failed IoU match, but still requires visual review. Loose exact means the text is right at IoU 0.25, but fails 0.5. Merged coverage is not counted as a missing detection. Partial/unmatched remains unresolved. Recognition errors have a matched box.</p>',
             '<p>Original statements: 27 variants of three layouts. Validation: eight variants of four layouts. Counts include repeated content across angles; these are synthetic diagnostics, not production accuracy estimates.</p>', table(stats,['family','group','truth']), '<h2>Ordinary single-character words</h2>',table(singles,['token','truth']),
             '<p>Digit coverage is sparse: only 1, 2 and 3 occur outside artificial character strips. This cannot establish general digit recall.</p>',
             '<h2>Cached dense-tile map trace</h2><p>For omitted ordinary words inside cached tile ownership, count pixels in the labelled word window before and after the existing 3×3 morphological opening. This is a diagnostic, not a tested recovery policy. Surviving pixels can still fail component/score filtering or ownership checks.</p><pre>'+html.escape(json.dumps(dict(Counter(t['stage'] for t in traces)),indent=2))+'</pre>',
             '<h2>Representative upright omissions</h2><p>Red: labelled word with no overlapping prediction. Context is the original raster. Examples prioritise short words and avoid repeated strings in the same layout/region.</p><div class="cards">']
    seen = set(); examples = []
    for p in result:
        if p['angle'] != 0:
            continue
        rows = sorted((w for w in p['rows'] if w['group']=='ordinary' and w['status']=='no_box_overlap'), key=lambda w:(len(w['text']),w['index']))
        for w in rows:
            key=(p['id'],w['region'],w['text'])
            if key in seen or len(examples)>=24:
                continue
            seen.add(key);examples.append((p,w))
    for p,w in examples:
        g=polygon(w['polygon']);x0,y0,x1,y1=g.bounds;W,H=p['width'],p['height']
        x=max(0,x0*W-180);y=max(0,y0*H-40);cw=min(W-x,(x1-x0)*W+360);ch=min(H-y,(y1-y0)*H+80)
        src=html.escape(os.path.relpath(p['image'],out).replace(os.sep,'/'),quote=True)
        points=' '.join(f'{a*W},{b*H}' for a,b in w['polygon'])
        parts.append(f'<div class="card"><b>{html.escape(w["text"])}</b> — {p["id"]} / {w["region"]}<svg viewBox="{x} {y} {cw} {ch}"><image href="{src}" width="{W}" height="{H}"/><polygon points="{points}" fill="none" stroke="#d52637" stroke-width="2"/></svg></div>')
    parts.append('</div><p><a href="audit.json">All word-level evidence</a></p>')
    (out/'index.html').write_text(''.join(parts),encoding='utf-8')
    print(json.dumps(dict(statistics=stats,singles=singles),indent=2))
    print('Cached tile trace:',dict(Counter(t['stage'] for t in traces)))
    print(out/'index.html')


if __name__=='__main__':
    main()
