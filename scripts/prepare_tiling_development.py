"""Freeze development pages for resolution/overlap experiments, separate from public tests."""
import gzip,hashlib,io,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_public_datasets import page_record,rect,write

def prepare(out):
    root=ROOT/'testdata/public'
    for name in ['funsd','cord-v2','hiertext']:
        folder=out/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'manifest.json').exists():continue
        pages=[]
        if name=='funsd':
            source=root/name/'raw/dataset/training_data'
            images=sorted((source/'images').glob('*.png'),key=lambda p:hashlib.sha256(p.stem.encode()).hexdigest())[:50]
            for image in images:
                data=json.loads((source/'annotations'/f'{image.stem}.json').read_text(encoding='utf-8'))
                words=[dict(text=w['text'],polygon=rect(w['box']),ignore=False) for b in data['form'] for w in b['words'] if w['text']]
                pages.append(page_record(name,'train',image,words))
        elif name=='cord-v2':
            import pyarrow.parquet as pq
            images=folder/'images';images.mkdir(exist_ok=True)
            for file in sorted((root/name/'data').glob('validation-*.parquet')):
                for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                    row=batch.to_pylist()[0];data=json.loads(row['ground_truth']);image=images/f'{len(pages):04d}.png'
                    with Image.open(io.BytesIO(row['image']['bytes'])) as im:im.convert('RGB').save(image)
                    words=[dict(text=w['text'],polygon=[[w['quad'][f'x{i}'],w['quad'][f'y{i}']] for i in range(1,5)],ignore=False) for line in data['valid_line'] for w in line['words'] if w['text']]
                    pages.append(page_record(name,'validation',image,words))
            assert len(pages)==100
        else:
            source=next((root/name/'source').glob('hiertext-*'))
            with gzip.open(source/'gt/validation.jsonl.gz','rt',encoding='utf-8') as f:data=json.load(f)
            records=sorted(data['annotations'],key=lambda r:hashlib.sha256(r['image_id'].encode()).hexdigest())[:200]
            images={p.stem:p for p in (root/name/'images/validation').rglob('*.jpg')}
            for record in records:
                words=[dict(text=w['text'],polygon=w['vertices'],ignore=not w['legible']) for para in record['paragraphs'] for line in para['lines'] for w in line['words']]
                pages.append(page_record(name,'validation',images[record['image_id']],words,image_id=record['image_id']))
        pages.sort(key=lambda p:p['id']);assert pages
        write(folder/'manifest.json',pages)
        print(name,len(pages),sum(not w['ignore'] for p in pages for w in p['words']),flush=True)
if __name__=='__main__':prepare(ROOT/'pybaseline/results/tiling_dev_v1')
