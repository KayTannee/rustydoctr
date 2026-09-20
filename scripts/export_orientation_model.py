"""Export docTR's cached page classifier to CPU ONNX; validate logits on CPU."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
import numpy as np
import onnxruntime as ort
import torch
from doctr.models import page_orientation_predictor


def main():
    torch.set_num_threads(1)
    predictor=page_orientation_predictor(pretrained=True).eval().cpu()
    model=predictor.model
    path=ROOT/'models/page_orientation.onnx'
    torch.manual_seed(1)
    sample=torch.rand(1,*model.cfg['input_shape'])
    with torch.inference_mode():
        torch.onnx.export(model,sample,str(path),input_names=['input'],output_names=['logits'],opset_version=17,dynamo=False)
        expected=model(sample).numpy()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    actual=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'input':sample.numpy()})[0]
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    metadata={k:model.cfg[k] for k in ['mean','std','classes','input_shape']}
    metadata.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),architecture='mobilenet_v3_small_page_orientation',scope='CPU page direction; class angles are clockwise input orientation, corrected counterclockwise')
    (ROOT/'models/page_orientation.json').write_text(json.dumps(metadata,indent=2))
    print('Exported CPU page classifier; Torch/ONNX logit parity passed')


if __name__=='__main__':main()
