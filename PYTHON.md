# Use Rust OCR from Python

For another Windows NVIDIA machine, use the transfer ZIP and
[deployment instructions](PYTHON_DEPLOYMENT.md). The wheel is version 0.2.0,
CPython 3.12+ / Windows x64. Run VS Code **Python: build transfer ZIP** to rebuild
the wheel, install it into the local baseline environment, and package models,
configs, examples and checksums. **OCR: build Python wheel** builds/installs only
the wheel. **Python: check installed CUDA stream** tests backpressure/shutdown.


Build/install the wheel into the existing benchmark environment:

Build-machine prerequisites: Rust MSVC toolchain, Visual Studio C++ Build Tools,
`uv`, and `.venv-baseline` with Python 3.12+ and `maturin>=1.9,<2`.
Install the build frontend if needed with
`uv pip install --python .venv-baseline/Scripts/python.exe "maturin>=1.9,<2"`.
The ZIP task also requires the exported models in `models/` and generated
`testdata/generated/a4_control.png`. These prerequisites apply to rebuilding,
not installing the transfer ZIP on another PC.

```powershell
./scripts/build_python.ps1 -Install
```

The distributable wheel is in `dist/`. It contains Rust plus the Python wrapper;
model weights, ONNX Runtime and CUDA/cuDNN DLLs remain external. The transfer ZIP
includes model weights; the `gpu` extra installs the NVIDIA runtime dependencies
without requiring Torch. This tested setup
uses Python 3.12+, ORT GPU 1.23.2 and separately installed CUDA 12/cuDNN 9 packages.
The wrapper finds ORT and uses its DLL preloader for pip-installed NVIDIA
runtimes or an existing compatible Torch installation, without importing Torch.
For another deployment, set `ORT_DYLIB_PATH` and, on Windows,
`RUSTYDOCTR_DLL_DIRS` (semicolon-separated). CUDA is required; no silent CPU fallback.

**Render PDFs, feed RGB pages, independently consume results:**

```powershell
./.venv-baseline/Scripts/python examples/stream_pdf.py testdata/generated/ocr_stress.pdf --config profiles/my-machine/config.json --output results.jsonl
```

The complete example is [examples/stream_pdf.py](examples/stream_pdf.py). Its core:

```python
from threading import Thread
from rustydoctr import Stream

with Stream(models="models", config="profiles/my-machine/config.json") as ocr:
    def feed():
        try:
            for page_id, width, height, rgb_bytes in rendered_pages():
                ocr.submit_rgb(page_id, width, height, rgb_bytes)
        finally:
            ocr.finish_input()

    producer = Thread(target=feed)
    producer.start()
    try:
        for result in ocr:
            handle_result(result)  # id, sequence, words
    finally:
        ocr.close()  # also unblocks feeding if the consumer fails
        producer.join()
```

The example's statistics include first-inference CUDA warmup. Use the benchmark
task for warmed steady-state throughput; a short cold run will look much slower.

Use tightly packed RGB `bytes`: `width * height * 3`, e.g. PyMuPDF `pix.samples`
or Pillow `image.convert("RGB").tobytes()`. Submission copies once into owned Rust
memory; no PNG encoding. Calls that initialize, submit, wait or close release the
GIL. Results preserve order and your `id`; each word includes normalized box
coordinates, text, confidence and detection objectness. `page` is reserved for the
native corpus benchmark; use `id` in applications.

`finish_input()` closes admission; continue consuming to drain. `close()` cancels
unfinished work and joins workers; an active CUDA operation must finish first.
`ocr.stats` is available after draining. The full example propagates producer errors
as well as inference/consumer errors.
Latency percentiles cover the most recent 4,096 results; the completion timeline
retains at most 8,192 entries, so long-lived streams do not grow statistics forever.
The saved `seconds` and `pages` fields control benchmarks only; a library stream
runs until `finish_input()` or cancellation.

Memory is bounded by configured in-flight pages, one queued input, one blocked
submission, bounded crop queues and an in-flight-sized result queue. A slow consumer
eventually blocks feeding. This bounds page counts, not bytes for arbitrarily huge
pages. Keep one producer and one consumer. Heavy Python postprocessing may need a
separate bounded process pool; the example's independent consumer writes JSONL.

This slice does **word OCR**, with optional [page orientation and fractional
deskew](PAGE_ORIENTATION.md). Experimental [line-guided crop alternatives](LINE_ORIENTATION.md)
are available with `config["line_guided_orientation"] = True`; they retain per-word
`crop_decision` diagnostics and default to off. General mixed local rotation,
line/block assembly and table/layout analysis remain incomplete.

Experimental dense-text refinement is opt-in after rebuilding the wheel:

```python
import json
config = json.load(open("profiles/my-machine/config.json"))
config["dense_refine"] = True
with Stream(models="models", config=config) as ocr:
    ...  # use the same independent producer/consumer pattern above
```

This selects at most one band (at most 25% of page height), detects two overlapping
1024 tiles, reconciles boxes, then recognizes once. Output `refinement_tiles`
records source-pixel crop bounds and center-ownership intervals. No selection
returns an empty list. Missing `dense_refine` defaults to false for old profiles.

Optional tile-only detector settings live in `config["dense_detection"]`.
See [the tuning results and configuration example](DENSE_TUNING.md); defaults
remain unchanged and full-page detection is unaffected.

Set `config["thin_recovery"] = True` to opt into bounded recovery of thin words
discarded by detector cleanup. It reuses the same streaming queues and models.
Accepted words carry a `thin_recovery` evidence object; the flag defaults to false.
See [native recovery controls and results](THIN_RECOVERY.md).
Dense refinement adds detector passes; thin recovery alone reuses existing maps.
These options add CPU work and host buffers; remeasure memory and throughput
before enabling it on a small GPU. It does not correct page/local rotation.

Thin recovery now includes one-hop line support for otherwise omitted narrow
characters. Dense refinement also reconciles seam ownership. Both share the
existing streaming queues and page admission limit; see [THIN_RECOVERY.md](THIN_RECOVERY.md).
