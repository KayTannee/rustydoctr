#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Use a Linux-native cache when building from a Windows filesystem in WSL.
cache="${RUSTYDOCTR_BUILD_CACHE:-$HOME/.cache/rustydoctr-build}"
export CARGO_HOME="${CARGO_HOME:-$cache/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$cache/rustup}"
export CARGO_TARGET_DIR="$cache/target"
export PATH="$CARGO_HOME/bin:$cache/venv/bin:$HOME/.local/bin:$PATH"
python_bin="${RUSTYDOCTR_BUILD_PYTHON:-$cache/venv/bin/python}"
"$python_bin" -m maturin build --release --locked --zig --target x86_64-unknown-linux-gnu --compatibility manylinux_2_28 --out dist
if [[ "${1:-}" == "--package" ]]; then
    "$python_bin" scripts/package_python.py --platform linux
fi
