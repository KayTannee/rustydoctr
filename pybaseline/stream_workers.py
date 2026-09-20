"""CPU-only feeder and result-writer processes for the streaming benchmark."""
import json
from multiprocessing import shared_memory
import os
import queue
import time


def feed(image_path, shm_name, shape, slots, free, ready, start, stop, seconds, output):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    shm=shared_memory.SharedMemory(name=shm_name)
    buffers=np.ndarray((slots,*shape),dtype=np.uint8,buffer=shm.buf)
    start.wait()
    began=time.perf_counter(); cpu_start=time.process_time(); deadline=began+seconds
    count=0; decode_seconds=slot_wait_seconds=0.0
    try:
        while not stop.is_set() and time.perf_counter()<deadline:
            waited=time.perf_counter()
            try:
                slot=free.get(timeout=.1)
            except queue.Empty:
                slot_wait_seconds+=time.perf_counter()-waited
                continue
            slot_wait_seconds+=time.perf_counter()-waited
            if time.perf_counter()>=deadline:
                free.put(slot); break
            admitted_ns=time.time_ns()
            decode_start=time.perf_counter()
            image=cv2.imread(image_path,cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode {image_path}')
            cv2.cvtColor(image,cv2.COLOR_BGR2RGB,dst=buffers[slot])
            decode_seconds+=time.perf_counter()-decode_start
            ready.put({'id':count,'slot':slot,'admitted_ns':admitted_ns,'ready_ns':time.time_ns()})
            count+=1
        ready.put(None)
        result={'pages':count,'wall_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu_start,
                'decode_copy_seconds':decode_seconds,'free_slot_wait_seconds':slot_wait_seconds,
                'mode':'Decode the same PNG from the warm filesystem cache for each submitted page; copy RGB into bounded shared-memory slots.'}
        with open(output,'w',encoding='utf-8') as file:
            json.dump(result,file,indent=2)
    finally:
        shm.close()


def consume(results, start, output):
    import numpy as np
    from pathlib import Path
    output=Path(output)
    start.wait()
    began=time.perf_counter(); cpu_start=time.process_time()
    latencies=[]; inference_to_post=[]; timeline=[]; count=0; expected=0
    serialize_seconds=wait_seconds=0.0; first_ns=None; last_ns=None
    with (output/'pages.jsonl').open('w',encoding='utf-8') as stream:
        while True:
            waited=time.perf_counter(); item=results.get(); wait_seconds+=time.perf_counter()-waited
            if item is None:
                break
            batch, pages=item
            if len(batch)!=len(pages):
                raise AssertionError(f'Result count mismatch: {len(batch)} inputs, {len(pages)} outputs')
            started=time.perf_counter()
            for meta,page in zip(batch,pages):
                if meta['id']!=expected:
                    raise AssertionError(f'Lost or reordered page: expected {expected}, received {meta["id"]}')
                stream.write(json.dumps({'sequence':meta['id'],'page':page},separators=(',',':'))+'\n')
                expected+=1
            stream.flush()
            finished_ns=time.time_ns()
            serialize_seconds+=time.perf_counter()-started
            for meta in batch:
                first_ns=meta['admitted_ns'] if first_ns is None else first_ns
                latencies.append((finished_ns-meta['admitted_ns'])/1e9)
                inference_to_post.append((finished_ns-meta['inference_complete_ns'])/1e9)
            count+=len(batch); last_ns=finished_ns
            timeline.append({'time_ns':finished_ns,'pages_total':count,'batch_pages':len(batch)})
        stream.flush(); os.fsync(stream.fileno())
    ended_ns=time.time_ns()
    result={'pages':count,'first_admitted_ns':first_ns,'last_completed_ns':last_ns,'durable_end_ns':ended_ns,
            'end_to_end_wall_seconds':(ended_ns-first_ns)/1e9 if first_ns else 0,
            'wall_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu_start,
            'serialize_flush_seconds':serialize_seconds,'input_wait_seconds':wait_seconds,
            'latency_seconds':{f'p{p}':float(np.percentile(latencies,p)) for p in [50,95,99]} if latencies else {},
            'post_delay_seconds_p95':float(np.percentile(inference_to_post,95)) if inference_to_post else None,
            'timeline':timeline,'output_bytes':(output/'pages.jsonl').stat().st_size,
            'latency_definition':'Admission before decode to post-process write+flush. Final fsync is included in aggregate makespan.'}
    (output/'post.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
