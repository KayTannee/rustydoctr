"""Download all public splits, verify archives, and retain dataset provenance."""
import argparse, concurrent.futures, hashlib, json, shutil, tarfile, time, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HIER_REV='70b6620b2b112597d8219e11eee9773a1403827c'

def read_url(url):
    with urllib.request.urlopen(url,timeout=60) as r:return json.load(r)

def download(url,path,sha=None):
    path.parent.mkdir(parents=True,exist_ok=True);meta=path.with_suffix(path.suffix+'.download.json')
    if path.exists() and meta.exists():
        info=json.loads(meta.read_text(encoding='utf-8'))
        if info['url']==url and (not sha or info['sha256']==sha):return path
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(4):
        try:
            offset=part.stat().st_size if part.exists() else 0
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req,timeout=90) as r:
                append=offset and r.status==206
                if append:assert r.headers['Content-Range'].startswith(f'bytes {offset}-')
                with part.open('ab' if append else 'wb') as f:
                    last=time.monotonic()
                    while chunk:=r.read(1024*1024):
                        f.write(chunk)
                        if time.monotonic()-last>20:print(path.name,round(f.tell()/2**20),'MiB',flush=True);last=time.monotonic()
            with part.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if sha and digest!=sha:raise ValueError(f'Checksum mismatch: {path}')
            part.replace(path);meta.write_text(json.dumps(dict(url=url,sha256=digest,bytes=path.stat().st_size),indent=2),encoding='utf-8')
            print('Downloaded',path.name,round(path.stat().st_size/2**20,1),'MiB',flush=True);return path
        except Exception:
            if attempt==3:raise
            time.sleep(2)

def extract(path,destination):
    done=destination/'.extracted.json'
    if done.exists():return
    destination.mkdir(parents=True,exist_ok=True)
    root=destination.resolve()
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                target=(root/item.filename).resolve()
                if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
            z.extractall(root)
    else:
        with tarfile.open(path) as t:t.extractall(root,filter='data')
    done.write_text(json.dumps(dict(archive=str(path.resolve()))),encoding='utf-8');print('Extracted',path.name,flush=True)

def funsd(out):
    p=download('https://guillaumejaume.github.io/FUNSD/dataset.zip',out/'funsd/dataset.zip','c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f');extract(p,out/'funsd/raw')

def cord(out):
    folder=out/'cord-v2';folder.mkdir(parents=True,exist_ok=True);pin=folder/'upstream.json'
    if pin.exists():info=json.loads(pin.read_text(encoding='utf-8'))
    else:
        revision=read_url('https://huggingface.co/api/datasets/naver-clova-ix/cord-v2')['sha']
        info=dict(revision=revision,files=read_url(f'https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/tree/{revision}?recursive=true'))
        pin.write_text(json.dumps(info,indent=2),encoding='utf-8')
    files=[f for f in info['files'] if f['path'].endswith('.parquet')]
    files.sort(key=lambda f:('test' not in f['path'],f['path']))
    for f in files:download(f"https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{info['revision']}/{f['path']}",folder/f['path'],f['lfs']['oid'])

def hiertext(out):
    folder=out/'hiertext';p=download(f'https://codeload.github.com/google-research-datasets/hiertext/zip/{HIER_REV}',folder/'source.zip');extract(p,folder/'source')
    for split in ['test','validation','train']:
        p=download(f'https://s3.amazonaws.com/open-images-dataset/ocr/{split}.tgz',folder/f'{split}.tgz');extract(p,folder/'images'/split)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'testdata/public');args=parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(fn,args.output):fn.__name__ for fn in [funsd,cord,hiertext]};failures=[]
        for job in concurrent.futures.as_completed(jobs):
            try:job.result();print('Complete:',jobs[job],flush=True)
            except Exception as e:failures.append((jobs[job],repr(e)));print('Failed:',jobs[job],repr(e),flush=True)
        if failures:raise RuntimeError(failures)
if __name__=='__main__':main()
