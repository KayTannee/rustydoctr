"""Independent process sampler: polling frequency is not NVML sensor resolution."""
import json
import multiprocessing as mp
import time
from pathlib import Path


def _sample(pid, output, interval, stop, ready):
    import psutil
    import pynvml as nv
    process = psutil.Process(pid)
    process.cpu_percent()
    psutil.cpu_percent(percpu=True)
    error = None
    try:
        nv.nvmlInit()
        handle = nv.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        handle, error = None, str(exc)
    with open(output, 'w', encoding='utf-8') as stream:
        ready.set()
        while not stop.is_set():
            start = time.perf_counter()
            try:
                row = {'time_ns': time.time_ns(), 'process_cpu_pct': process.cpu_percent(),
                       'system_cpu_per_core': psutil.cpu_percent(percpu=True),
                       'rss_bytes': process.memory_info().rss, 'nvml_error': error}
                if handle is not None:
                    def read(fn):
                        try:
                            return fn()
                        except nv.NVMLError:
                            return None
                    util = read(lambda: nv.nvmlDeviceGetUtilizationRates(handle))
                    row.update(gpu_util_pct=util.gpu if util else None,
                               gpu_memory_controller_pct=util.memory if util else None,
                               device_vram_bytes=read(lambda: nv.nvmlDeviceGetMemoryInfo(handle).used),
                               power_mw=read(lambda: nv.nvmlDeviceGetPowerUsage(handle)),
                               sm_clock_mhz=read(lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)))
                stream.write(json.dumps(row)+'\n'); stream.flush()
            except psutil.NoSuchProcess:
                break
            stop.wait(max(0, interval-(time.perf_counter()-start)))
    if handle is not None:
        nv.nvmlShutdown()


class Monitor:
    def __init__(self, path, interval=0.05):
        import os
        self.path = Path(path)
        ctx = mp.get_context('spawn')
        self.stop, self.ready = ctx.Event(), ctx.Event()
        self.process = ctx.Process(target=_sample, args=(os.getpid(), str(path), interval, self.stop, self.ready))

    def __enter__(self):
        self.process.start()
        if not self.ready.wait(30):
            self.process.terminate()
            raise RuntimeError('Resource sampler failed to start')
        return self

    def __exit__(self, *args):
        self.stop.set(); self.process.join(10)
        if self.process.is_alive():
            self.process.terminate(); self.process.join()

    def summary(self):
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        result = {'samples': len(rows), 'nvml_error': next((r['nvml_error'] for r in rows if r['nvml_error']), None)}
        for key in ('process_cpu_pct', 'gpu_util_pct', 'gpu_memory_controller_pct', 'device_vram_bytes', 'rss_bytes', 'power_mw'):
            values = [r[key] for r in rows if r.get(key) is not None]
            result[key+'_mean'] = sum(values)/len(values) if values else None
            result[key+'_max'] = max(values) if values else None
        return result
