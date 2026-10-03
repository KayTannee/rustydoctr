#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${PYTHON:-python3.12}"
if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv .venv
fi
.venv/bin/python verify_bundle.py
.venv/bin/python -m pip install -c runtime-lock.txt './rustydoctr-0.2.0-cp312-abi3-manylinux_2_28_x86_64.whl[gpu,examples]'
.venv/bin/python -m rustydoctr doctor --models models --config configs/low-vram.json --smoke
