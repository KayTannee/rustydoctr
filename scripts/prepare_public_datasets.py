"""Build frozen test manifests from original word annotations; never resize images."""
import argparse, gzip, hashlib, io, json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def rect(b):
    x0,y0,x1,y1=b;return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
def page_record(dataset,split,image,words,**extra):
    with Image.open(image) as im:w,h=im.size
    for word in words:word['polygon']=[[float(x)/w,float(y)/h] for x,y in word['polygon']]
    return dict(id=f'{dataset}_{split}_{image.stem}',dataset=dataset,split=split,image=str(image.resolve()),width=w,height=h,words=words,**extra)
def prepare(root,name):
    out=root/name/'prepared';pages=[]
    if name=='funsd':
        folder=root/name/'raw/dataset/testing_data'
        for image in sorted((folder/'images').glob('*.png')):
            data=json.loads((folder/'annotations'/f'{image.stem}.json').read_text(encoding='utf-8'))
            words=[dict(text=w['text'],polygon=rect(w['box']),region=b['label'],ignore=False) for b in data['form'] for w in b['words'] if len(w['text'])>0]
            pages.append(page_record(name,'test',image,words))
        assert len(pages)==50
    elif name=='cord-v2':
        import pyarrow.parquet as pq
        images=out/'images';images.mkdir(parents=True,exist_ok=True)
        index=0
        for file in sorted((root/name/'data').glob('test-*.parquet')):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                row=batch.to_pylist()[0];raw=row['image']['bytes'];data=json.loads(row['ground_truth']);image=images/f'{index:04d}.png'
                with Image.open(io.BytesIO(raw)) as im:im.convert('RGB').save(image)
                write(out/'annotations'/f'{index:04d}.json',data)
                words=[dict(text=w['text'],polygon=[[w['quad'][f'x{i}'],w['quad'][f'y{i}']] for i in range(1,5)],region=line.get('category',''),ignore=False) for line in data['valid_line'] for w in line['words'] if len(w['text'])>0]
                pages.append(page_record(name,'test',image,words));index+=1
        assert len(pages)==100
    elif name=='hiertext':
        source=next((root/name/'source').glob('hiertext-*'))
        with gzip.open(source/'gt/test.jsonl.gz','rt',encoding='utf-8') as f:data=json.load(f)
        images={p.stem:p for p in (root/name/'images/test').rglob('*.jpg')}
        for record in data['annotations']:
            words=[dict(text=w['text'],polygon=w['vertices'],ignore=not w['legible'],region='word',handwritten=w.get('handwritten',line.get('handwritten',False)),vertical=w.get('vertical',line.get('vertical',False))) for para in record['paragraphs'] for line in para['lines'] for w in line['words']]
            page=page_record(name,'test',images[record['image_id']],words,image_id=record['image_id'])
            assert [page['width'],page['height']]==[record['image_width'],record['image_height']]
            pages.append(page)
        assert len(pages)==1634
    pages.sort(key=lambda p:p['id'])
    write(out/'manifest.json',pages);write(out/'workload.json',dict(pages=[dict(id=p['id'],image=p['image']) for p in pages]))
    files=[p for p in (root/name).rglob('*.download.json')]+[out/'manifest.json']
    write(out/'provenance.json',{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print(name,len(pages),'pages',sum(len(p['words']) for p in pages),'words',sum(sum(not w['ignore'] for w in p['words']) for p in pages),'legible',flush=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'testdata/public');parser.add_argument('--datasets',nargs='+',default=['funsd','cord-v2','hiertext']);a=parser.parse_args()
    for name in a.datasets:prepare(a.root,name)
if __name__=='__main__':main()
