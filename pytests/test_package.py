"""CPU checks for portable wrapper validation; no runtime imports on package import."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release_wrapper', Path(__file__).resolve().parents[1]/'python/rustydoctr/__init__.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

def test_configs_are_independent():
    a=wrapper.default_config();a['dense_detection']['bin_thresh']=.9
    assert wrapper.default_config()['dense_detection']['bin_thresh']==.3
    assert not a['dense_refine'] and not a['thin_recovery']
    assert wrapper.default_config('low-vram')['inflight']==1
    with pytest.raises(ValueError):wrapper.default_config('typo')

def test_unknown_options_fail_before_native_loading(tmp_path):
    with pytest.raises(ValueError,match='Unknown configuration'):
        wrapper.Stream(models=tmp_path,config={'reco_bach':128})

def test_missing_models_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError,match='db_resnet34.onnx'):
        wrapper.Stream(models=tmp_path)
