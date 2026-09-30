"""Bounded, ordered RGB page streaming into the Rust CUDA pipeline."""
import importlib.util
from functools import cache
import json
import os
from pathlib import Path

__version__ = "0.2.0"
_dll_handles = []


def default_config(profile="balanced"):
    """Return a fresh conservative starting config, not a measured machine profile."""
    config = dict(size=1024, reco_batch=128, det_batch=1, workers=2, inflight=2,
                  arena_mib=6144, det_arena_mib=4096, vram_limit_mib=0,
                  seconds=0, pages=0, page_orientation=False, deskew=False,
                  dense_refine=False, thin_recovery=False, line_guided_orientation=False,
                  dense_detection=dict(bin_thresh=0.3, box_thresh=0.1, unclip_ratio=1.5))
    if profile == "low-vram":
        config.update(reco_batch=64, inflight=1, arena_mib=3072, det_arena_mib=2048,
                      vram_limit_mib=3584)
    elif profile != "balanced":
        raise ValueError("profile must be 'balanced' or 'low-vram'")
    return config



@cache
def _runtime():
    """Locate an externally installed ORT/CUDA runtime; wheels do not bundle it."""
    paths = []
    spec = importlib.util.find_spec("onnxruntime")
    if spec and spec.submodule_search_locations:
        capi = Path(next(iter(spec.submodule_search_locations))) / "capi"
        name = "onnxruntime.dll" if os.name == "nt" else "libonnxruntime.so.1.23.2"
        if (capi / name).exists():
            os.environ.setdefault("ORT_DYLIB_PATH", str(capi / name))
        paths.append(capi)
    spec = importlib.util.find_spec("torch")
    if spec and spec.submodule_search_locations:
        paths.append(Path(next(iter(spec.submodule_search_locations))) / "lib")
    paths.extend(Path(p) for p in os.environ.get("RUSTYDOCTR_DLL_DIRS", "").split(os.pathsep) if p)
    # cuDNN loads additional engine DLLs lazily, beyond ORT's primary preloads.
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            paths.extend(sorted(Path(root).glob("*/bin")))
    if os.name == "nt":
        for path in paths:
            if path.is_dir():
                _dll_handles.append(os.add_dll_directory(str(path)))
                os.environ["PATH"] = str(path) + os.pathsep + os.environ["PATH"]
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('Install the GPU runtime: pip install "rustydoctr[gpu]"') from exc
    # Load pip-installed NVIDIA runtimes (or an existing compatible Torch runtime).
    # No Torch import or Python inference session is required.
    if os.name == "nt":
        ort.preload_dlls()



class Stream:
    """One producer and one consumer; submit blocks when queues are full.

    ``submit_rgb`` copies immutable, tightly packed RGB bytes. Models remain
    resident until close. Finish input, then iterate until exhaustion to drain.
    Always use a context manager so consumer failures cancel blocked producers.
    """

    def __init__(self, models="models", config=None):
        if config is None:
            config = default_config()
        if isinstance(config, (str, os.PathLike)):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        supplied = dict(config)
        self.config = default_config()
        unknown = supplied.keys() - self.config.keys()
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        self.config.update(supplied)
        models = Path(models).resolve()
        required = ["metadata.json", "db_resnet34.onnx", "parseq.onnx"]
        if self.config["page_orientation"]:
            required += ["page_orientation.json", "page_orientation.onnx"]
        missing = [name for name in required if not (models / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model files in {models}: {', '.join(missing)}")
        _runtime()
        from ._native import RawStream
        self._raw = RawStream(str(Path(models).resolve()), json.dumps(self.config))

    def submit_rgb(self, id, width, height, data):
        """Submit tightly packed RGB ``bytes``; releases the GIL while blocked."""
        self._raw.submit_rgb(str(id), width, height, data)

    def finish_input(self):
        self._raw.finish_input()

    def __iter__(self):
        return self

    def __next__(self):
        result = self._raw.recv_json()
        if result is None:
            raise StopIteration
        return json.loads(result)

    @property
    def stats(self):
        """Final statistics, available after draining. Latencies retain 4096 pages."""
        return json.loads(self._raw.stats_json())

    def close(self):
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
