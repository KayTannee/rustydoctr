# Upstream performance review

Compared the installed docTR **1.1.0** with upstream tagged **0.12.0** source on
2026-09-18. This is source inspection, not a timed historical A/B comparison.

| Area | 0.12.0 | 1.1.0 | Likely implication, not measured speedup |
|---|---|---|---|
| OCR segmentation maps | Requested and binarized unconditionally | Requested only for orientation/straightening | Avoids unused maps and associated work on straight-page runs |
| Rotated box scoring | Allocates a full-page mask per contour | Allocates a mask limited to the contour's bounding region | Much less per-box memory work on pages containing many rotated words |
| Detector postprocessing | Iterates samples serially | Uses the shared thread-pool helper across samples | CPU postprocessing can overlap across a detector batch |
| Thread-pool sizing | Up to 16 workers even for small inputs | Limits workers to item count; bypasses pool for one item | Reduces pool startup overhead for small batches |
| Page straightening | Detect, rotate, detect again | Still detect, rotate, detect again | The extra detection pass remains |

The rotated-score computation also differs numerically: the older code divides by
nonzero probability products, while the newer code averages values selected by the
polygon mask. Do not assume bitwise/threshold equivalence from an optimization claim.

Preprocessing already used threads in 0.12.0. Current FAST predictors reparameterize
their model at construction. These are reasons to profile the actual installed
version rather than assume every host-side operation is serialized by the GIL.

Sources:

- [0.12 OCR predictor](https://github.com/mindee/doctr/blob/v0.12.0/doctr/models/predictor/pytorch.py)
- [1.1 OCR predictor](https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/predictor/pytorch.py)
- [0.12 detector postprocessing](https://github.com/mindee/doctr/blob/v0.12.0/doctr/models/detection/core.py)
- [1.1 detector postprocessing](https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/detection/core.py)
- [0.12 thread helper](https://github.com/mindee/doctr/blob/v0.12.0/doctr/utils/multithreading.py)
- [1.1 thread helper](https://github.com/mindee/doctr/blob/v1.1.0/doctr/utils/multithreading.py)
- [1.1 FAST factory](https://github.com/mindee/doctr/blob/v1.1.0/doctr/models/detection/zoo.py)

The [release history](https://github.com/mindee/doctr/releases) documents PyTorch-only
support from 1.0, an updated CRNN-VGG16 checkpoint, an orientation-condition fix,
and 1.1's optional layout/table models and original-coordinate remapping. Older
release notes also describe `torch.compile`; this baseline uses eager float32.
Compilation, AMP, ONNX, TensorRT and tiling have not been benchmarked in this sweep.

For monitoring, [NVIDIA's NVML definition](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html)
describes activity windows of about 1/6 to 1 second, depending on device. A 50 ms
poll interval therefore does not imply 50 ms independent GPU activity measurements.
GPU activity is not an SM occupancy or FLOPS-efficiency metric.
