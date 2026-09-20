"""Measure the installed Python wheel: Python decode/feed + Rust OCR + Python JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
import time

from PIL import Image
from rustydoctr import Stream
from pybaseline.monitor import Monitor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--workload', type=Path, default=Path('testdata/throughput.json'))
    p.add_argument('--models', default='models')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--pages', type=int, default=0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.config.read_text())
    pages = json.loads(a.workload.read_text())['pages']
    go, stop = Event(), Event()
    errors = []
    started = {}
    load = time.perf_counter()
    with Monitor(a.output/'telemetry.jsonl') as monitor, Stream(a.models, config) as stream:
        load_seconds = time.perf_counter() - load
        warm = time.perf_counter()

        def submit(index):
            with Image.open(pages[index % len(pages)]['image']) as source:
                rgb = source.convert('RGB')
                stream.submit_rgb(str(index), rgb.width, rgb.height, rgb.tobytes())

        def feed():
            try:
                for i in range(len(pages)):
                    submit(i)
                while not go.wait(.1):
                    if stop.is_set():
                        return
                i = 0
                while not stop.is_set():
                    if a.pages and i >= a.pages:
                        break
                    if not a.pages and i and i % len(pages) == 0 and time.perf_counter()-started['time'] >= a.seconds:
                        break
                    submit(i)
                    i += 1
            except BaseException as exc:
                errors.append(exc)
            finally:
                stream.finish_input()

        producer = Thread(target=feed, name='rgb-producer')
        producer.start()
        try:
            for i in range(len(pages)):
                result = next(stream)
                assert result['id'] == str(i) and result['sequence'] == i
            warm_seconds = time.perf_counter()-warm
            print('Warmup complete; measuring installed Python library', flush=True)
            count = words = 0
            first = {}
            timeline = []
            start_ns = time.time_ns()
            started['time'] = time.perf_counter()
            go.set()
            with (a.output/'pages.jsonl').open('w', encoding='utf-8') as output:
                for result in stream:
                    assert result['sequence'] == count+len(pages) and result['id'] == str(count)
                    result['sequence'] = count
                    result['page'] = count % len(pages)
                    output.write(json.dumps(result, separators=(',', ':'))+'\n')
                    output.flush()
                    first.setdefault(str(result['page']), result)
                    count += 1
                    words += len(result['words'])
                    timeline.append([time.perf_counter()-started['time'], count])
                    if count % len(pages) == 0:
                        print(count, 'pages', round(timeline[-1][0], 1), 's', flush=True)
                output.flush()
                os.fsync(output.fileno())
            wall = time.perf_counter()-started['time']
            end_ns = time.time_ns()
            native = stream.stats
        finally:
            stop.set()
            stream.close()
            producer.join()
        if errors:
            raise errors[0]
    summary = dict(implementation='rust_python_binding', config=config, pages=count, words=words,
                   wall_seconds=wall, pages_per_second=count/wall, start_ns=start_ns, end_ns=end_ns,
                   model_load_seconds=load_seconds, warmup_seconds=warm_seconds,
                   corpus_sha256=hashlib.sha256(a.workload.read_bytes()).hexdigest(),
                   first_pages=first, timeline=timeline, max_inflight=native['max_inflight'],
                   resources=monitor.summary(), native_including_warmup=native,
                   scope='Python PNG decode/RGB copy + Rust word OCR + Python JSON parsing/writing/fsync. Full-corpus warmup excluded.')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'DONE {count/wall:.3f} pages/s ({count} pages)', flush=True)


if __name__ == '__main__':
    main()
