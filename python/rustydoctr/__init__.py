"""Bounded, ordered RGB page streaming into the Rust CUDA pipeline."""
import importlib.util
from functools import cache
import json
import os
from pathlib import Path

_dll_handles = []


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
    if os.name == "nt":
        for path in paths:
            if path.is_dir():
                _dll_handles.append(os.add_dll_directory(str(path)))
                os.environ["PATH"] = str(path) + os.pathsep + os.environ["PATH"]


class Stream:
    """One producer and one consumer; submit blocks when queues are full.

    ``submit_rgb`` copies immutable, tightly packed RGB bytes. Models remain
    resident until close. Finish input, then iterate until exhaustion to drain.
    Always use a context manager so consumer failures cancel blocked producers.
    """

    def __init__(self, models="models", config=None):
        if config is None:
            raise ValueError("Pass a measured config.json or a configuration dict")
        if isinstance(config, (str, os.PathLike)):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        self.config = dict(config)
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
