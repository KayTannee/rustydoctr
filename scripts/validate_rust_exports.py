"""Check dynamic exported shapes and logits against export-mode PyTorch on CPU."""
import os
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
import numpy as np
import torch
import onnxruntime as ort
from doctr.models import db_resnet34,parseq

def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
    rows=[]
    for name,factory,shapes in [('db_resnet34',db_resnet34,[(1,3,64,64),(1,3,96,64)]),
                               ('parseq',parseq,[(1,3,32,128),(3,3,32,128)])]:
        model=factory(pretrained=True,exportable=True).eval()
        session=ort.InferenceSession(str(ROOT/'models'/f'{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        for shape in shapes:
            x=torch.rand(shape)
            with torch.inference_mode(): expected=model(x)['logits'].numpy()
            actual=session.run(None,{'input':x.numpy()})[0]
            np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-4)
            rows.append(dict(model=name,input_shape=shape,output_shape=actual.shape,
                             max_abs=float(np.abs(actual-expected).max()),mean_abs=float(np.abs(actual-expected).mean())))
            print(rows[-1],flush=True)
    (ROOT/'models/export_validation.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
