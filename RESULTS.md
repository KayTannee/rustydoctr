# Current throughput result

RTX 5070 Ti 16 GB, Ryzen 7 8700G. FP32 DB ResNet34 + PARSeq, detector 1536.
Ten identical upright synthetic pages (including dense A3), repeated in order.
Loading and a full-corpus warmup are excluded; decode, OCR, drain and JSONL are timed.

| Run | Pages/s | Measured pages | Seconds |
|---|---:|---:|---:|
| Python/docTR, PyTorch (earlier baseline) | 0.671 | 210 | 312.8 |
| Python, same ONNX graphs (earlier baseline) | 0.728 | 230 | 315.8 |
| Rust native, 512 crops, 4 admitted pages (fresh run) | 1.154 | 360 | 312.0 |
| Installed Python → Rust library, same settings | 1.158 | 360 | 310.8 |
| Rust 4 GB target proxy, 128 crops, 2 admitted pages | 0.928 | 290 | 312.5 |

The Python interface retained native throughput: **0.4% difference**, within normal
single-run variation. Python performed PNG decoding, RGB submission, JSON parsing
and output writing, using independent producer/consumer threads. No PNG re-encoding.
All ten first-pass word lists matched the native reference (16,447 detected words).

The 4 GB target kept both models resident and used **3.37 GiB peak incremental
device VRAM** above a 2.21 GiB desktop baseline. This was a budgeted test on the
16 GB desktop, not validation on the actual 4 GB laptop. Reprofile there.

The earlier controlled Rust/Python comparison showed roughly 48% more throughput
than Python using the same ONNX graphs, or 61% over docTR/PyTorch, with 512 crops.
The fresh runs followed a reboot and ran somewhat faster; do not attribute that
difference to the bindings. Backend, preprocessing and scheduling all differ from
docTR; these measurements do not isolate the GIL or the language alone.

Open `pybaseline/results/throughput/report.html` for the complete audit, accuracy,
GPU/VRAM traces and original runs. The current local parameters are
`profiles/desktop-verified/config.json`. Follow [BENCHMARK.md](BENCHMARK.md) to
generate a new profile, and [PYTHON.md](PYTHON.md) to use it.

This is upright word OCR. PDF rasterization is demonstrated separately, and
orientation/retries/layout are deferred. Dense A3 remains an accuracy stress case.
