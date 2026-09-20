"""Convenient repeatable launcher for the VS Code GPU-Z throughput task."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--page', choices=['upright_characters', 'skew7_characters',
                                         'skew7_mixed_characters'], default='upright_characters')
    parser.add_argument('--seconds', type=int, choices=[300, 600], default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root/'testdata/rotation/manifest.json'
    if not manifest.exists():
        parser.error('Generate pages first: python scripts/generate_rotation_pdfs.py')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output = root/'pybaseline/results/manual_streaming'/f'{stamp}-{args.page}-{uuid4().hex[:8]}'
    command = [sys.executable, '-u', '-m', 'pybaseline.stream',
               '--manifest', str(manifest), '--page', args.page,
               '--det', 'db_resnet34', '--reco', 'parseq', '--rotation', 'crops',
               '--size', '1536', '--batch', '8', '--reco-batch', '512',
               '--threads', '1', '--seconds', str(args.seconds), '--output', str(output)]
    print(f'DB ResNet34 + PARSeq | 1536 | page batch 8 / recognition batch 512\n'
          f'Feeding for {args.seconds}s after loading and warmup, then draining queued pages.\n'
          f'Watch GPU-Z now. Progress appears every 80 pages during measurement.\n'
          f'Results: {output}\n', flush=True)
    return subprocess.call(command, cwd=root)


if __name__ == '__main__':
    raise SystemExit(main())
