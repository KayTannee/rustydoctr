# Install on Linux

Use `rustydoctr-0.2.0-linux-x64.zip` for Linux x86-64, glibc 2.28+, 64-bit
CPython 3.12+ and an NVIDIA GPU with a CUDA 12-compatible driver. Alpine/musl,
ARM and AMD GPUs are not supported by this build. No Rust, Torch, docTR or
CUDA Toolkit is needed on the receiving machine.

Extract the ZIP, open a terminal in its folder, then run:

```bash
bash install.sh
.venv/bin/python examples/stream_images.py samples/document.png --config configs/low-vram.json --output results.jsonl
.venv/bin/python examples/stream_pdf.py /path/document.pdf --config configs/low-vram.json --dpi 300 --output pdf-results.jsonl
```

Install Python's venv support first if your distribution packages it separately
(on Ubuntu 24.04: `sudo apt install python3.12-venv`). To select another supported
interpreter, run `PYTHON=python3.13 bash install.sh`.
The installer downloads pinned Linux ONNX Runtime and NVIDIA CUDA/cuDNN packages
and runs a GPU smoke test. Internet and several GB of free disk space are needed.
The NVIDIA driver must already work (`nvidia-smi`). Under WSL2, use the Windows
NVIDIA driver; do not install a Linux display driver inside WSL.

For an existing environment:

```bash
python -m pip install -c runtime-lock.txt './rustydoctr-0.2.0-cp312-abi3-manylinux_2_28_x86_64.whl[gpu,examples]'
python -m rustydoctr doctor --models models --config configs/low-vram.json --smoke
```

The Python API and examples match Windows. Supply tightly packed RGB bytes to
`Stream.submit_rgb`; use one producer thread and consume results independently.
The queues apply backpressure. Keep the stream alive across pages to amortize
model loading and first-inference initialization. The first page can take about
a minute. `low-vram.json` is a starting point, not a guarantee for every 4 GB GPU;
device usage includes desktop and driver memory outside the model arenas.
Quality refinements are experimental and disabled by default. Heatmap fusion,
layout and table analysis are not part of this streaming package.
Native per-process CPU/RSS telemetry is currently Windows-only; OCR stage timing
and NVIDIA device monitoring remain available on Linux.

Verify the copied bundle with `python verify_bundle.py`. Keep `models/` beside
the examples or pass its absolute path with `--models`.

## Rebuild in the repository

VS Code tasks: **Python: build Linux wheel** and **Python: build Linux transfer ZIP**.
On Windows they use the existing WSL distribution named `Ubuntu`; on Linux they
use Bash directly. Commands run from the repository root:

```bash
bash scripts/build_python_linux.sh
bash scripts/build_python_linux.sh --package
```

Build prerequisites are Rust, Python 3.12+, a C compiler, and a Python build
environment containing `maturin[zig]` and `patchelf`. The default isolated tool
cache is `$HOME/.cache/rustydoctr-build`: Rust uses its `cargo/` and `rustup/`
directories, and the Python build interpreter is `venv/bin/python`.
Set `CARGO_HOME`, `RUSTUP_HOME` and `RUSTYDOCTR_BUILD_PYTHON` to use existing tools.
Zig targets the manylinux 2.28 ABI; the wheel is audited during the build.
Packaging also needs the repository's exported models and generated sample PNG.

## Validation

Tested on Ubuntu 24.04 / WSL2, Python 3.12.3, RTX 5070 Ti, driver 616.92,
ORT 1.23.2 and the pinned CUDA/cuDNN packages, without Torch or docTR.
All 29 Rust unit tests passed, as did CUDA smoke, streaming order/backpressure,
cancellation and VRAM-guard checks. Three repeated sample pages each returned
900 words. Other distributions and physical 4 GB cards have not been tested.
