# Third-party implementation notices

The straight DB postprocessing and preprocessing/decoding conventions in this
vertical slice are adapted from docTR 1.1.0, Copyright (C) 2021–2026 Mindee,
licensed under Apache License 2.0. The upstream license is reproduced in
`licenses/doctr-LICENSE`.

Sources: `doctr/models/detection/core.py`,
`doctr/models/detection/differentiable_binarization/base.py`,
`doctr/models/detection/_utils/base.py`, `doctr/utils/geometry.py`,
`doctr/models/preprocessor/pytorch.py`,
`doctr/models/recognition/predictor/_utils.py`,
`doctr/models/recognition/utils.py`, and
`doctr/models/recognition/parseq/pytorch.py` at
https://github.com/mindee/doctr/tree/v1.1.0.

Changes: Rust reimplementation, connected-component bounds in place of external
contour bounds for the upright path, analytical axis-aligned rectangle expansion,
native resize and JSON output, and wide-crop splitting/string remapping in the
bounded pipeline. Rotation and layout are not implemented. Model weights are exported locally from the installed docTR checkpoint
cache and are not committed. Cargo.lock records the Rust dependency versions.
