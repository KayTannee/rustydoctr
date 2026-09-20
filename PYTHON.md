# Use Rust OCR from Python

Build/install the wheel into the existing benchmark environment:

```powershell
./scripts/build_python.ps1
```

The distributable wheel is in `dist/`. It contains Rust plus the Python wrapper;
model weights, ONNX Runtime and CUDA/cuDNN DLLs remain external. This tested setup
uses Python 3.12+, ORT GPU 1.23.2 and CUDA 12/cuDNN 9 from the baseline environment.
The wrapper finds ORT and Torch's DLL directories without importing Torch.
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

This slice does upright **word OCR**. Rotation correction, retry passes, line/block
assembly and table/layout analysis are not implemented yet.
