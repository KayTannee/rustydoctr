"""Capture hardware/driver and idle device state without collecting other app names."""
import json
from pathlib import Path
import platform
import time


def main():
    import psutil
    import pynvml as nv
    import torch
    nv.nvmlInit()
    handle=nv.nvmlDeviceGetHandleByIndex(0)
    memory=nv.nvmlDeviceGetMemoryInfo(handle)
    cpu=platform.processor()
    if platform.system()=='Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    result={'captured_ns':time.time_ns(),'cpu':cpu,'physical_cores':psutil.cpu_count(logical=False),
            'logical_cores':psutil.cpu_count(),'ram_bytes':psutil.virtual_memory().total,
            'gpu':nv.nvmlDeviceGetName(handle),'driver':nv.nvmlSystemGetDriverVersion(),
            'device_total_vram_bytes':memory.total,'device_used_vram_at_capture_bytes':memory.used,
            'cuda_compute_capability':torch.cuda.get_device_capability(),
            'note':'Snapshot may be taken during experiments; used VRAM is not an idle baseline.'}
    nv.nvmlShutdown()
    output=Path('pybaseline/results/environment.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
