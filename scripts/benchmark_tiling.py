"""Accuracy experiment using the Rust Python library. Full tiling is an experimental Python adapter."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
from threading import Thread
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_public import points,score_run
from scripts.prepare_public_datasets import write

MODES={'plain1024':(1024,False,None),'plain1536':(1536,False,None),'selective1024':(1024,True,None),
       'tiles1536_o128':(1024,False,(1536,128,0)),
       'tiles2048_o0':(1024,False,(2048,0,0)),
       'tiles2048_o128':(1024,False,(2048,128,0)),
       'tiles2048_o256':(1024,False,(2048,256,0)),
       'tiles2048_o128_shift':(1024,False,(2048,128,.5))}

def windows(width,height,longside,overlap,shift):
    # Crop at original resolution; native preprocessing maps the square to 1024.
    side=max(1,round(max(width,height)*1024/longside))
    stride=max(1,round(side*(1024-overlap)/1024));offset=round(stride*shift)
    starts=lambda length:list(range(-offset,length,stride))
    return [(x,y,x+side,y+side) for y in starts(height) for x in starts(width)]

def mapped(word,box,width,height):
    x,y,x1,y1=box;side=x1-x
    q=points(word)*side+[x,y]
    cx,cy=q.mean(axis=0)
    if not (0<=cx<width and 0<=cy<height):return None
    # Prefer observations with context on every interior crop edge, without using labels/text.
    local=points(word)*side
    margins=[]
    if x>0:margins.append(local[:,0].min())
    if y>0:margins.append(local[:,1].min())
    if x1<width:margins.append(side-local[:,0].max())
    if y1<height:margins.append(side-local[:,1].max())
    q[:,0]=np.clip(q[:,0],0,width);q[:,1]=np.clip(q[:,1],0,height)
    return dict(word,polygon=(q/[width,height]).tolist(),quadrilateral=(q/[width,height]).tolist(),tile_box=box,edge_margin=float(min(margins,default=side)/side))

def merge(words):
    # Class/text agnostic IoU NMS. Keep raw observations so other merge policies can be audited.
    ordered=sorted(words,key=lambda w:(w['edge_margin'],w.get('objectness',0)),reverse=True)
    kept=[];bounds=[]
    for word in ordered:
        p=points(word);a=np.r_[p.min(axis=0),p.max(axis=0)];area=np.prod(np.maximum(0,a[2:]-a[:2]));duplicate=False
        for b in bounds:
            inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
            union=area+np.prod(np.maximum(0,b[2:]-b[:2]))-inter
            if union>0 and inter/union>=.5:duplicate=True;break
        if not duplicate:kept.append(word);bounds.append(a)
    return kept

def run(pages,folder,mode,arena_mib=6144):
    from rustydoctr import Stream
    size,select,tile=MODES[mode]
    config=dict(vram_limit_mib=0,seconds=300.0,pages=0,size=size,det_batch=1,reco_batch=256,workers=2,inflight=3,arena_mib=arena_mib,det_arena_mib=4096,page_orientation=False,deskew=False,dense_refine=select,thin_recovery=False,line_guided_orientation=False)
    folder.mkdir(parents=True,exist_ok=True)
    errors=[];started=time.perf_counter();crops=0
    with Stream(models=ROOT/'models',config=config) as stream:
        # A single producer and consumer retain bounded native backpressure.
        def feed():
            try:
                for i,page in enumerate(pages):
                    with Image.open(page['image']) as im:image=im.convert('RGB')
                    boxes=windows(image.width,image.height,*tile) if tile else [(0,0,image.width,image.height)]
                    for j,(x,y,x1,y1) in enumerate(boxes):
                        if tile:
                            crop=Image.new('RGB',(x1-x,y1-y),'white');crop.paste(image,(-x,-y))
                        else:crop=image
                        stream.submit_rgb(f'{i}:{j}',crop.width,crop.height,crop.tobytes())
            except BaseException as e:errors.append(e)
            finally:stream.finish_input()
        producer=Thread(target=feed);producer.start()
        try:
            with (folder/'pages.jsonl').open('w',encoding='utf-8') as output, (folder/'raw_tiles.jsonl').open('w',encoding='utf-8') as raw:
                observations=[]
                for result in stream:
                    i,j=map(int,result['id'].split(':'));page=pages[i];W,H=page['width'],page['height'];boxes=windows(W,H,*tile) if tile else [(0,0,W,H)];crops+=1
                    if tile:
                        raw.write(json.dumps(dict(page=page['id'],tile=j,box=boxes[j],words=result['words']))+'\n')
                        observations.extend(w for word in result['words'] if (w:=mapped(word,boxes[j],W,H)) is not None)
                    else:observations=result['words']
                    if j==len(boxes)-1:
                        words=merge(observations) if tile else observations
                        output.write(json.dumps(dict(id=page['id'],sequence=i,words=words,tiles=boxes if tile else [],refinement_tiles=result.get('refinement_tiles',[])))+'\n');output.flush();observations=[]
                        if (i+1)%25==0:print(mode,i+1,'/',len(pages),flush=True)
        finally:stream.close();producer.join()
        if errors:raise errors[0]
        stats=stream.stats
    seconds=time.perf_counter()-started
    write(folder/'summary.json',dict(mode=mode,pages=len(pages),crop_passes=crops,wall_seconds=seconds,pages_per_second=len(pages)/seconds,config=config,tile_config=tile,stats=stats,timing_note='Cold exploratory elapsed time includes loading, recognition per tile and Python merging; not production throughput.'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'pybaseline/results/tiling_dev_v1');p.add_argument('--modes',nargs='+',default=list(MODES));p.add_argument('--datasets',nargs='+',default=['funsd','cord-v2','hiertext']);p.add_argument('--arena-mib',type=int,default=6144);p.add_argument('--score-only',action='store_true');a=p.parse_args()
    for ds in a.datasets:
        manifest=a.root/ds/'manifest.json';pages=json.loads(manifest.read_text(encoding='utf-8'))
        for mode in a.modes:
            folder=a.root/ds/mode
            if not a.score_only and not (folder/'summary.json').exists():
                print('START',ds,mode,flush=True);run(pages,folder,mode,a.arena_mib)
                write(folder/'provenance.json',dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            if (folder/'summary.json').exists() and not (folder/'scores.json').exists():score_run(manifest,folder)
if __name__=='__main__':main()

