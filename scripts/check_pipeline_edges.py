"""CUDA integration checks for blank-page backpressure and failure termination."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy()
runtime=ROOT/'.venv-baseline/Lib/site-packages/onnxruntime/capi'
env['ORT_DYLIB_PATH']=str(runtime/'onnxruntime.dll')
env['PATH']=str(runtime)+os.pathsep+str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib')+os.pathsep+env['PATH']
(ROOT/'.cache').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='pipeline-edges-',dir=ROOT/'.cache') as temp:
    folder=Path(temp);image=folder/'blank.png';Image.new('RGB',(64,64),'white').save(image)
    workload=folder/'workload.json'
    workload.write_text(json.dumps({'pages':[{'id':'blank','image':str(image)}]}))
    command=[str(ROOT/'target/release/throughput.exe'),'--workload',str(workload),'--size','64','--inflight','1','--det-batch','2','--reco-batch','128','--pages','3','--arena-mib','2048']
    subprocess.run(command+['--output',str(folder/'blank-results')],cwd=ROOT,env=env,check=True,timeout=90)
    summary=json.loads((folder/'blank-results/summary.json').read_text())
    assert summary['pages']==3 and summary['words']==0 and summary['max_inflight']==1
    corrupt=folder/'corrupt.png';corrupt.write_bytes(image.read_bytes()[:50])
    workload.write_text(json.dumps({'pages':[{'id':'blank','image':str(image)},{'id':'corrupt','image':str(corrupt)}]}))
    failed=subprocess.run(command+['--output',str(folder/'bad-results')],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert failed.returncode!=0 and not (folder/'bad-results/summary.json').exists()
    print('PASS: one-slot blank-page drain; corrupt-input failure terminates without a false success summary')
    print(failed.stderr[-1500:])
