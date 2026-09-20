"""CUDA integration checks for the installed wheel (run separately from benchmarks)."""
import json
from pathlib import Path
from threading import Thread
import time
from rustydoctr import Stream

CONFIG = dict(size=64, reco_batch=32, det_batch=2, workers=2, inflight=1,
              arena_mib=2048, det_arena_mib=512, vram_limit_mib=0, seconds=0, pages=0)
WHITE = bytes([255]) * (64 * 64 * 3)


def main():
    import rustydoctr
    assert 'site-packages' in str(Path(rustydoctr.__file__)), 'Test the installed wheel'
    submitted, errors = [], []
    with Stream(config=CONFIG) as stream:
        try:
            stream.submit_rgb('bad', 64, 64, b'x')
            raise AssertionError('Malformed RGB accepted')
        except ValueError:
            pass

        def feed():
            try:
                for i in range(20):
                    stream.submit_rgb(str(i), 64, 64, WHITE)
                    submitted.append(i)
            except BaseException as exc:
                errors.append(exc)
            finally:
                stream.finish_input()

        producer = Thread(target=feed, daemon=True)
        producer.start()
        time.sleep(2)
        assert len(submitted) <= 5, f'Unbounded producer: {len(submitted)}'
        count = 0
        for record in stream:
            assert record['id'] == str(count) and record['sequence'] == count and record['words'] == []
            count += 1
            time.sleep(.02)
        producer.join(5)
        assert not producer.is_alive() and not errors and count == 20
        assert stream.stats['max_inflight'] == 1 and stream.stats['pages'] == 20
        try:
            stream.submit_rgb('late', 64, 64, WHITE)
            raise AssertionError('Submission after finish accepted')
        except RuntimeError:
            pass
    print('PASS: malformed RGB, ordered blank pages, slow consumer, bounded producer, drain', flush=True)

    with Stream(config=CONFIG) as stream:
        stream.finish_input()
        assert list(stream) == [] and stream.stats['pages'] == 0
    print('PASS: empty input', flush=True)

    with Stream(config=CONFIG) as stream:
        stopped = []
        def blocked():
            try:
                for i in range(100):
                    stream.submit_rgb(str(i), 64, 64, WHITE)
            except RuntimeError:
                stopped.append(True)
            finally:
                stream.finish_input()
        producer = Thread(target=blocked, daemon=True)
        producer.start()
        time.sleep(1)
        stream.close()
        producer.join(10)
        assert not producer.is_alive() and stopped
    print('PASS: cancellation unblocks a producer without consuming results', flush=True)
    try:
        with Stream(config=dict(CONFIG, vram_limit_mib=1)) as stream:
            stream.submit_rgb('budget', 64, 64, WHITE)
            stream.finish_input()
            list(stream)
        raise AssertionError('Insufficient VRAM budget accepted')
    except RuntimeError as exc:
        assert 'VRAM exceeded' in str(exc), str(exc)
    print('PASS: insufficient VRAM budget is reported to Python', flush=True)
    # Exercise rendering and result handling through the actual example separately.
    Path('pybaseline/results/throughput/python_stream_checks.json').write_text(json.dumps({'passed': True}))


if __name__ == '__main__':
    main()
