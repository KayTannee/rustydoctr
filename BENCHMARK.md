# Benchmark and profile

Run from the repository root in PowerShell. Close other GPU workloads.

**First setup on a new Windows/NVIDIA machine** (Rust, `uv`, Python 3.12):

```powershell
./pybaseline/setup.ps1
./scripts/setup_rust.ps1
./scripts/build_python.ps1
```

**Generate standard samples, tune this machine, benchmark, save parameters:**

VS Code → **Run Task → OCR: profile system and Python library**.
Choose `--auto` or `--vram-4gb`, then 300 or 600 seconds per measured run.
Calibration and warmup add several minutes; native and Python runs are sequential.

Equivalent command (omit `--vram-4gb` for the normal profile):

```powershell
./.venv-baseline/Scripts/python scripts/profile_system.py --vram-4gb --python-check --output profiles/my-machine
```

Open `profiles/my-machine/report.html`. Reuse its **config.json** in the
[Python example](PYTHON.md). `profile.json` records hardware, idle VRAM, image/model
hashes and measured throughput. The task prints the timestamped output location.

**Repeat a benchmark with saved settings**, without recalibrating:

```powershell
./.venv-baseline/Scripts/python scripts/profile_system.py --config profiles/my-machine/config.json --python-check --output profiles/recheck
```

Use a fresh output directory each time. Compare machines only when image/model
hashes match. All tests use the same ten upright pages, repeated for at least five
minutes; loading and one full-corpus warmup are excluded. Decode, inference, queue
drain and JSONL writing are included. PDF rendering and orientation are excluded.

The selected setting is the lowest-memory candidate within 3% of the fastest
eligible short trial, followed by a sustained run. It is a measured starting point,
not an exhaustive optimum. The 4 GB mode keeps both FP32 models resident, accounts
for currently free VRAM, and uses smaller batches; it does not quantize or lower
detector resolution. Sampled device-memory guards can miss brief peaks.

For baseline docTR commands and detailed findings, see [THROUGHPUT.md](THROUGHPUT.md).
