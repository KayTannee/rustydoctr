"""CPU-only alternate merging of cached full-tile observations; no label-based decisions."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_tiling import MODES,mapped,merge,windows
from scripts.score_public import points,score_run
from scripts.prepare_public_datasets import write

def guarded(words,tiles,width,height):
    kept=[]
    for word in words:
        a,b,c,d=word['tile_box'];side=c-a
        if word['edge_margin']>8/1024:kept.append(word);continue
        q=points(word)*[width,height];x0,y0=q.min(axis=0);x1,y1=q.max(axis=0)
        # Only delegate an edge observation where another tile offers >=16 detector-pixel context.
        alternate=any((x,y,xx,yy)!=(a,b,c,d) and x0>=x+(16/1024*(xx-x) if x>0 else 0) and y0>=y+(16/1024*(yy-y) if y>0 else 0) and x1<=xx-(16/1024*(xx-x) if xx<width else 0) and y1<=yy-(16/1024*(yy-y) if yy<height else 0) for x,y,xx,yy in tiles)
        if not alternate:kept.append(word)
    return merge(kept)

def main():
    root=ROOT/'pybaseline/results/tiling_dev_v1'
    for ds in ['funsd','cord-v2','hiertext']:
        pages=json.loads((root/ds/'manifest.json').read_text(encoding='utf-8'));byid={p['id']:p for p in pages}
        for mode,(_,_,tile) in MODES.items():
            source=root/ds/mode
            if tile is None or tile[1]==0 or not (source/'summary.json').exists():continue
            folder=root/ds/(mode+'_guarded');folder.mkdir(exist_ok=True)
            if (folder/'scores.json').exists():continue
            grouped={p['id']:[] for p in pages}
            for line in (source/'raw_tiles.jsonl').open(encoding='utf-8'):
                r=json.loads(line);p=byid[r['page']]
                grouped[p['id']].extend(w for word in r['words'] if (w:=mapped(word,r['box'],p['width'],p['height'])) is not None)
            with (folder/'pages.jsonl').open('w',encoding='utf-8') as f:
                for i,p in enumerate(pages):
                    tiles=windows(p['width'],p['height'],*tile)
                    words=guarded(grouped[p['id']],tiles,p['width'],p['height'])
                    f.write(json.dumps(dict(id=p['id'],sequence=i,words=words,tiles=tiles))+'\n')
            summary=json.loads((source/'summary.json').read_text(encoding='utf-8'));summary.update(mode=folder.name,wall_seconds=None,pages_per_second=None,timing_note='Reuses raw inference from '+mode+'; no independent runtime.',merge_policy='8-pixel edge guard, 16-pixel alternate context, then IoU 0.5 NMS')
            write(folder/'summary.json',summary);score_run(root/ds/'manifest.json',folder)
if __name__=='__main__':main()
