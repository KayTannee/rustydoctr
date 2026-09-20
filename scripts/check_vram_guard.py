"""Ensure the sampled VRAM guard cancels work and records a clear failure."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='vram-guard-', dir=ROOT/'.cache') as temp:
    folder = Path(temp)
    assert folder.resolve().parent == (ROOT/'.cache').resolve()
    result = subprocess.run([
        'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(ROOT/'scripts/run_pipeline.ps1'), '--output', str(folder/'run'),
        '--workload', str(ROOT/'testdata/throughput.json'), '--pages', '1',
        '--size', '1536', '--arena-mib', '3072', '--det-arena-mib', '2048',
        '--reco-batch', '32', '--inflight', '2', '--vram-limit-mib', '1024',
    ], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0, 'An intentionally insufficient memory target must fail'
    failure = json.loads((folder/'run/failure.json').read_text())
    assert failure['vram_budget_exceeded'] and 'VRAM exceeded' in failure['error']
    assert not (folder/'run/summary.json').exists()
    print('PASS: memory target violation stops the pipeline, with no false success summary')
    print(failure['error'])
