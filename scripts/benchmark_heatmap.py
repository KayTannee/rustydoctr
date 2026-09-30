"""Run native heatmap fusion on the same frozen pages and exact crop grids as box merging."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_tiling import MODES,windows
from scripts.prepare_public_datasets import write
from scripts.score_public import score_run
from scripts.tune_dense_detection import environment

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',default=['funsd','cord-v2','hiertext']);p.add_argument('--modes',nargs='+',default=['tiles1536_o128','tiles2048_o128','tiles2048_o256','tiles2048_o128_shift']);a=p.parse_args();root=ROOT/'pybaseline/results/tiling_dev_v1'
    for ds in a.datasets:
        manifest=root/ds/'manifest.json';pages=json.loads(manifest.read_text(encoding='utf-8'))
        decoded={}
        for page in pages:
            image=Path(page['image'])
            if image.suffix.lower() in ['.jpg','.jpeg']:
                cache=root/ds/'decoded_rgb';cache.mkdir(exist_ok=True);target=cache/(page['id']+'.png')
                if not target.exists():
                    with Image.open(image) as im:im.convert('RGB').save(target)
                decoded[page['id']]=str(target)
            else:decoded[page['id']]=str(image)
        for mode in a.modes:
            tile=MODES[mode][2];folder=root/ds/(mode+'_heatmap');folder.mkdir(exist_ok=True);jobs=[dict(id=p['id'],image=decoded[p['id']],tiles=windows(p['width'],p['height'],*tile)) for p in pages];write(folder/'jobs.json',jobs)
            if not (folder/'summary.json').exists():
                print('HEATMAP',ds,mode,flush=True);subprocess.run([str(ROOT/'target/release/heatmap_probe.exe'),'--jobs',str(folder/'jobs.json'),'--output',str(folder)],cwd=ROOT,env=environment(),check=True)
            # Validate identical per-tile detector boxes on the first page against the existing library run.
            controls=[]
            with (root/ds/mode/'raw_tiles.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line)
                    if r['page']!=pages[0]['id']:break
                    actual=json.loads((folder/f'control_tile_{r["tile"]}.json').read_text(encoding='utf-8'))
                    assert len(actual)==len(r['words']),(ds,mode,r['tile'],len(actual),len(r['words']))
                    maxdiff=0.
                    for x,y in zip(actual,r['words']):
                        assert np.allclose(x['polygon'],y['polygon'],atol=1e-6)
                        maxdiff=max(maxdiff,abs(x['objectness']-y['objectness']))
                    assert maxdiff<1e-5;controls.append(dict(tile=r['tile'],boxes=len(actual),objectness_max_difference=maxdiff))
            write(folder/'controls.json',controls)
            write(folder/'provenance.json',dict(binary_sha256=hashlib.sha256((ROOT/'target/release/heatmap_probe.exe').read_bytes()).hexdigest(),jobs_sha256=hashlib.sha256((folder/'jobs.json').read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
            shared=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            for policy in ['uniform','weighted']:
                run=folder/policy;write(run/'summary.json',dict(mode=mode+'_heatmap_'+policy,pages=len(pages),crop_passes=shared['detector_passes'],wall_seconds=None,shared_probe_seconds=shared['wall_seconds'],policy=policy,timing_note=shared['note']))
                if not (run/'scores.json').exists():score_run(manifest,run)
if __name__=='__main__':main()
