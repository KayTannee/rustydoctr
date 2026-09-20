"""Convert all sampler logs beneath a directory to flat CSV for external plotting."""
import argparse
import csv
import json
from pathlib import Path


def convert(path):
    rows=[]
    for line in path.read_text().splitlines():
        row=json.loads(line)
        for i,value in enumerate(row.pop('system_cpu_per_core', [])):
            row[f'system_cpu_core_{i}_pct']=value
        rows.append(row)
    if not rows:
        return
    start=rows[0]['time_ns']
    for row in rows:
        row['elapsed_seconds']=(row['time_ns']-start)/1e9
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix('.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path,nargs='?',default=Path('pybaseline/results'))
    args=p.parse_args()
    for path in args.root.rglob('telemetry.jsonl'):
        convert(path)
