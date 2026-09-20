"""Generate the standard corpus, calibrate CUDA OCR, and save a reusable config."""
import argparse
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def fingerprint():
    workload = json.loads((ROOT/'testdata/throughput.json').read_text())['pages']
    images = {p['id']: hashlib.sha256(Path(p['image']).read_bytes()).hexdigest() for p in workload}
    models = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'models').glob('*.onnx'))}
    # Independent of drive letters and checkout paths.
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT/'src').rglob('*.rs'))}
    sources['Cargo.lock'] = hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest()
    return dict(images=images, models=models, rust_sources=sources,
                corpus_sha256=hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest())


def machine():
    import psutil
    import pynvml as nv
    nv.nvmlInit()
    try:
        device = nv.nvmlDeviceGetHandleByIndex(0)
        memory = nv.nvmlDeviceGetMemoryInfo(device)
        return dict(os=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                    ram_bytes=psutil.virtual_memory().total, gpu=nv.nvmlDeviceGetName(device),
                    driver=nv.nvmlSystemGetDriverVersion(), total_vram_bytes=memory.total,
                    initial_device_vram_bytes=memory.used, initial_free_vram_bytes=memory.free,
                    python=sys.version, packages={name: importlib.metadata.version(name) for name in
                                                 ['onnxruntime-gpu', 'Pillow', 'PyMuPDF', 'reportlab']})
    finally:
        nv.nvmlShutdown()


def native(config, output, seconds):
    args = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            ROOT/'scripts/run_pipeline.ps1', '--output', output, '--seconds', seconds]
    for key, value in config.items():
        if key not in ('seconds', 'pages'):
            args += ['--'+key.replace('_', '-'), value]
    run(*args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'profiles'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--vram-4gb', action='store_true')
    p.add_argument('--auto', action='store_true', help='Default calibration mode (for the VS Code task)')
    p.add_argument('--python-check', action='store_true', help='Also measure the installed wheel with the selected config')
    p.add_argument('--config', type=Path, help='Benchmark saved parameters without recalibrating')
    p.add_argument('--import-run', type=Path, help='Package an existing native run; does not rerun or claim current hardware')
    a = p.parse_args()
    if a.seconds <= 0:
        p.error('--seconds must be positive')
    if a.import_run and (a.config or a.vram_4gb):
        p.error('--import-run cannot be combined with tuning options')
    if a.config and a.vram_4gb:
        p.error('Use --vram-4gb when calibrating; a saved config already specifies its budget')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    if not a.import_run:
        # Always regenerate from fixed seed/DPI, then select the same ten upright pages.
        run(sys.executable, ROOT/'scripts/generate_test_pdfs.py', '--output', ROOT/'testdata/generated')
    run(sys.executable, ROOT/'scripts/prepare_throughput.py')
    if a.import_run:
        folder = a.import_run.resolve()
        summary = json.loads((folder/'summary.json').read_text())
        hardware = {'imported_run': str(folder), 'note': 'Historical run; hardware provenance remains with its original report.'}
    else:
        hardware = machine()
        if a.config:
            native(json.loads(a.config.read_text()), a.output/'native', a.seconds)
            folder = a.output/'native'
        else:
            run('powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ROOT/'scripts/run_pipeline.ps1',
                '--output', a.output/'native', '--seconds', a.seconds, '--vram-4gb' if a.vram_4gb else '--auto')
            folder = a.output/'native/run'
        summary = json.loads((folder/'summary.json').read_text())
    config = summary['config'].copy()
    # Support pre-memory-guard historical results, without mutating their evidence.
    config.setdefault('det_arena_mib', config['arena_mib']//4)
    config.setdefault('vram_limit_mib', 0)
    config.update(pages=0, seconds=a.seconds)
    config_path = a.output/'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(), hardware=hardware,
                    mode='imported' if a.import_run else ('reused' if a.config else 'calibrated'),
                    config_source=str(a.config.resolve()) if a.config else None,
                    fingerprints=fingerprint(), native_summary=str(folder/'summary.json'),
                    native_pages_per_second=summary['pages_per_second'], native_pages=summary['pages'],
                    native_seconds=summary['wall_seconds'], resources=summary['resources'], config=config,
                    selection='Best measured eligible candidate within 3% of fastest, preferring lower memory; or explicitly reused/imported parameters.')
    if a.python_check:
        run(sys.executable, '-u', '-m', 'pybaseline.throughput_binding', '--config', config_path,
            '--output', a.output/'python', '--seconds', a.seconds)
        binding = json.loads((a.output/'python/summary.json').read_text())
        evidence['python_pages_per_second'] = binding['pages_per_second']
        evidence['python_over_native'] = binding['pages_per_second']/summary['pages_per_second']
        reference = summary['first_pages']
        evidence['python_same_words'] = all(
            [w['text'] for w in binding['first_pages'][key]['words']] == [w['text'] for w in value['words']]
            for key, value in reference.items())
        if not evidence['python_same_words']:
            raise RuntimeError('Python/native word outputs differ; inspect saved results before using this profile')
    (a.output/'profile.json').write_text(json.dumps(evidence, indent=2))
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>' for k, v in config.items())
    (a.output/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width"><title>OCR system profile</title>
    <style>body{{font:16px system-ui;max-width:950px;margin:40px auto;padding:20px}}td{{padding:6px 20px;border-bottom:1px solid #ddd}}pre{{overflow:auto}}</style>
    <h1>OCR system profile</h1><p>Native: <b>{summary['pages_per_second']:.3f} pages/s</b>, {summary['pages']} pages in {summary['wall_seconds']:.1f}s after full-corpus warmup.</p>
    <p>Use <a href="config.json">config.json</a> with the Python Stream API. Complete hardware, model/corpus hashes and measurements: <a href="profile.json">profile.json</a>.</p>
    <table>{rows}</table><p>FP32 DB ResNet34 + PARSeq, upright word OCR at 1536. Both models stay resident. Idle device memory is included in calibration. The 4 GB setting is an incremental allocation target, not a guarantee on an untested GPU. Arena caps and sampled guards can miss transient allocations.</p>
    <p>Compare systems only when corpus/model hashes match. Initialization and full-corpus warmup are excluded. Decoding, queue drain and output writing are included. This profiles tested batch candidates; it does not find a global optimum or continuously retune.</p>
    <pre>{html.escape(json.dumps(evidence, indent=2))}</pre></html>''', encoding='utf-8')
    print(f'Configuration: {config_path}\nReport: {a.output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
