import json
import queue
import threading
import time
import pytest
from pybaseline.stream_workers import consume


def test_writer_preserves_every_page_and_sequence(tmp_path):
    channel=queue.Queue(); start=threading.Event(); start.set()
    now=time.time_ns()
    meta=[{'id':i,'admitted_ns':now-1000000,'inference_complete_ns':now} for i in range(3)]
    channel.put((meta[:2],[{'blocks':[]},{'blocks':[]}]))
    channel.put((meta[2:],[{'blocks':[]}]))
    channel.put(None)
    consume(channel,start,str(tmp_path))
    records=[json.loads(line) for line in (tmp_path/'pages.jsonl').read_text().splitlines()]
    assert [r['sequence'] for r in records]==[0,1,2]
    result=json.loads((tmp_path/'post.json').read_text())
    assert result['pages']==3
    assert result['latency_seconds']['p95']>=0


def test_writer_rejects_missing_or_reordered_pages(tmp_path):
    channel=queue.Queue(); start=threading.Event(); start.set()
    channel.put(([{'id':1,'admitted_ns':time.time_ns(),'inference_complete_ns':time.time_ns()}],[{'blocks':[]}]))
    with pytest.raises(AssertionError,match='Lost or reordered'):
        consume(channel,start,str(tmp_path))


def test_writer_rejects_missing_results(tmp_path):
    channel=queue.Queue(); start=threading.Event(); start.set()
    channel.put(([{'id':0}],[]))
    with pytest.raises(AssertionError,match='Result count mismatch'):
        consume(channel,start,str(tmp_path))
