# Install on another Windows PC

Copy and extract the entire `rustydoctr-0.2.0-windows-x64.zip`.
Requires Windows x64, 64-bit CPython 3.12+ (tested on 3.12), and an NVIDIA GPU
with a driver compatible with the installed CUDA 12 runtime. No Rust compiler,
PyTorch, docTR, CUDA Toolkit or repository checkout is needed on the target PC.

From PowerShell in the extracted folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\.venv\Scripts\python.exe examples\stream_images.py samples\document.png --config configs\low-vram.json --output results.jsonl
.\.venv\Scripts\python.exe examples\stream_pdf.py C:\path\document.pdf --config configs\low-vram.json --dpi 300 --output pdf-results.jsonl
```

The installer creates `.venv` and downloads pinned ONNX Runtime/CUDA/cuDNN and
example dependencies from PyPI. Internet is required for that step. NVIDIA
runtime dependencies are large (roughly 2 GB of downloads); they are not in the
transfer ZIP. An NVIDIA driver must already be installed. If Windows reports a
missing `VCRUNTIME`/`MSVCP` DLL, install Microsoft's Visual C++ 2015–2022 x64
Redistributable. This wheel is Windows-only; use the separate
`rustydoctr-0.2.0-linux-x64.zip` bundle for Linux.

Manual installation into your own existing environment:

```powershell
python -m pip install -c runtime-lock.txt ".\rustydoctr-0.2.0-cp312-abi3-win_amd64.whl[gpu,examples]"
python -m rustydoctr doctor --models models --config configs\low-vram.json --smoke
```

Avoid installing CPU `onnxruntime` alongside `onnxruntime-gpu` in the same venv.
`doctor --smoke` actually loads both OCR models and runs CUDA inference; merely
listing CUDA as an available provider does not prove that its DLLs load.
The first page can take about a minute while CUDA/cuDNN initializes kernels;
keep one `Stream` alive across pages to amortize that cost. A short cold run is
not a throughput benchmark.

## Call from your pipeline

```python
from threading import Thread
from rustydoctr import Stream, default_config

config = default_config("low-vram")
with Stream(models=r"C:\path\rustydoctr\models", config=config) as ocr:
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
            handle_result(result)  # id, sequence, words, optional geometry
    finally:
        ocr.close()
        producer.join()
```

Supply tightly packed RGB bytes (three bytes per pixel), not encoded PNG/JPEG
bytes. The examples propagate producer exceptions, drain results and shut down
cleanly. Submission and reception release the GIL. Queues bound how far feeding
can run ahead; a slow consumer applies backpressure. Use separate processes for
heavy Python result processing when appropriate.

## Configuration and scope

`low-vram.json` is a conservative starting point for 4 GB cards: 1024 detector,
recognition batch 64, one admitted page, 3 GiB combined ORT arenas and a 3.5 GiB
idle-relative guard. Arena caps are not total-device memory caps; desktop usage,
other processes and driver allocations still matter. This is not a guarantee
that arbitrary pages fit a 4 GB card. Try smaller recognition batches if needed.
`balanced.json` uses larger arenas/batches for GPUs with more free memory.
Neither profile claims to be the optimal config for the other machine.

Word boxes/text/confidence and ordered streaming are supported. Page orientation
and fractional deskew are optional (`page_orientation=true`, `deskew=true`).
Dense refinement, thin recovery and line-orientation alternatives are experimental
and off by default because public tests showed regressions. Heatmap fusion remains
an offline experiment; it is not exposed by this streaming release. Layout/table
analysis and docTR's full document-object API are not included.

For OCR output, polygons/quadrilaterals are normalized to the original image.
Keep `models/` beside the examples, or pass its absolute path. Verify copied files
with `python verify_bundle.py`; checksums cover models, wheel and support files.

Runtime installation reference:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
