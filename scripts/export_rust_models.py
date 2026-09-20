"""Export the existing docTR weights for the Rust CUDA vertical slice."""
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
import torch
import doctr
from doctr.models import db_resnet34, parseq


def main():
    torch.set_num_threads(4)
    out=ROOT/'models'; out.mkdir(exist_ok=True)
    metadata={'doctr':doctr.__version__,'torch':torch.__version__, 'opset':17,
              'scope':'Upright word OCR only. PARSeq export uses its full-length decoding path.'}
    for name, factory, shape in [('db_resnet34',db_resnet34,(1,3,1024,1024)),('parseq',parseq,(2,3,32,128))]:
        model=factory(pretrained=True,exportable=True).eval()
        path=out/f'{name}.onnx'
        if not path.exists():
            print('Exporting',name,flush=True)
            dynamic={'input':{0:'batch'},'logits':{0:'batch'}}
            if name=='db_resnet34':
                dynamic['input'].update({2:'height',3:'width'})
                dynamic['logits'].update({2:'height',3:'width'})
            with torch.inference_mode():
                torch.onnx.export(model,torch.rand(shape),str(path),input_names=['input'],
                                  output_names=['logits'],dynamic_axes=dynamic,opset_version=17,dynamo=False)
        metadata[name]={'mean':model.cfg['mean'],'std':model.cfg['std'],
                        'input_shape':model.cfg['input_shape'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if name=='parseq':metadata[name]['vocab']=model.vocab
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Export complete',flush=True)


if __name__=='__main__':main()
