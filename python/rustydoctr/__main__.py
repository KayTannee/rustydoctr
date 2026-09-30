"""Environment diagnostics and a small real-CUDA smoke check."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from . import Stream, __version__, default_config, _runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['doctor'])
    p.add_argument('--models', default='models')
    p.add_argument('--config')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    report = dict(rustydoctr=__version__, python=sys.version, platform=platform.platform(),
                  models=str(Path(args.models).resolve()))
    try:
        report['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used',
             '--format=csv,noheader'], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report['nvidia_smi'] = str(exc)
    try:
        _runtime()
        import onnxruntime as ort
        report.update(onnxruntime=ort.__version__, providers=ort.get_available_providers(),
                      ort_library=os.environ.get('ORT_DYLIB_PATH'))
        if args.smoke:
            config = args.config or default_config('low-vram')
            with Stream(models=args.models, config=config) as stream:
                stream.submit_rgb('smoke', 64, 64, bytes([255]) * (64*64*3))
                stream.finish_input()
                rows = list(stream)
                if len(rows) != 1 or rows[0]['id'] != 'smoke':
                    raise RuntimeError('Smoke test output mismatch')
                report.update(smoke='passed', pages=stream.stats['pages'], words=len(rows[0]['words']))
        print(json.dumps(report, indent=2))
    except Exception as exc:
        report['error'] = str(exc)
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

if __name__ == '__main__':
    main()
