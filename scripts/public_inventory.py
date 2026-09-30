"""Audit full downloaded split counts and archive provenance without loading images."""
import json
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'testdata/public';result={}
    fun={s:len(list((root/'funsd/raw/dataset'/folder/'images').glob('*.png'))) for s,folder in [('train','training_data'),('test','testing_data')]};assert fun==dict(train=149,test=50)
    cord={s:sum(pq.ParquetFile(p).metadata.num_rows for p in (root/'cord-v2/data').glob(s+'-*.parquet')) for s in ['train','validation','test']};assert cord==dict(train=800,validation=100,test=100)
    hier={s:len(list((root/'hiertext/images'/s).rglob('*.jpg'))) for s in ['train','validation','test']};assert hier==dict(train=8281,validation=1724,test=1634)
    for name,counts in [('funsd',fun),('cord-v2',cord),('hiertext',hier)]:
        result[name]=dict(splits=counts,archives=[dict(file=str(p.relative_to(root)),**json.loads(p.read_text(encoding='utf-8'))) for p in (root/name).rglob('*.download.json')])
    out=ROOT/'pybaseline/results/public_v1/download_inventory.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v['splits'] for k,v in result.items()});print('Total archive GiB',sum(x['bytes'] for v in result.values() for x in v['archives'])/2**30)
if __name__=='__main__':main()
