"""Assemble a platform-specific transfer bundle with verified local models."""
import argparse,hashlib,importlib.util,json,shutil,zipfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION='0.2.0'
def build(staging, platform='windows'):
    tag='win_amd64' if platform=='windows' else 'manylinux_2_28_x86_64'
    wheel=ROOT/'dist'/f'rustydoctr-{VERSION}-cp312-abi3-{tag}.whl'
    if not wheel.exists():raise SystemExit('Build the Python wheel first.')
    with zipfile.ZipFile(wheel) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert 'rustydoctr/__main__.py' in z.namelist()
    out=staging/f'rustydoctr-{VERSION}-{platform}-x64';out.mkdir(parents=True,exist_ok=True)
    def copy(source,destination):
        target=out/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    copy(wheel.relative_to(ROOT),wheel.name)
    for name in ['db_resnet34.onnx','parseq.onnx','metadata.json','page_orientation.onnx','page_orientation.json']:copy('models/'+name,'models/'+name)
    meta=json.loads((out/'models/metadata.json').read_text(encoding='utf-8'))
    orient=json.loads((out/'models/page_orientation.json').read_text(encoding='utf-8'))
    for model,expected in [(name,meta[name]['sha256']) for name in ['db_resnet34','parseq']]+[('page_orientation',orient['sha256'])]:
        assert hashlib.sha256((out/'models'/f'{model}.onnx').read_bytes()).hexdigest()==expected,model
    spec=importlib.util.spec_from_file_location('release_config',ROOT/'python/rustydoctr/__init__.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (out/'configs').mkdir(exist_ok=True)
    for profile in ['balanced','low-vram']:(out/'configs'/f'{profile}.json').write_text(json.dumps(module.default_config(profile),indent=2),encoding='utf-8')
    for name in ['stream_images.py','stream_pdf.py']:copy('examples/'+name,'examples/'+name)
    guide='PYTHON_DEPLOYMENT.md' if platform=='windows' else 'LINUX_DEPLOYMENT.md'
    copy('docs/guides/'+guide,'README.md');copy('docs/legal/THIRD_PARTY.md','THIRD_PARTY.md');copy('licenses/doctr-LICENSE','licenses/doctr-LICENSE')
    ext='ps1' if platform=='windows' else 'sh'
    copy('scripts/install_transfer.'+ext,'install.'+ext);copy('scripts/verify_transfer.py','verify_bundle.py')
    lock='requirements-deploy-lock.txt' if platform=='windows' else 'requirements-deploy-linux-lock.txt'
    copy('pybaseline/'+lock,'runtime-lock.txt')
    copy('testdata/generated/a4_control.png','samples/document.png')
    readme='Generated local sample from the repository OCR fixtures. No public benchmark images or customer data are included.\n'
    (out/'samples/README.txt').write_text(readme,encoding='utf-8')
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='checksums.json')
    (out/'checksums.json').write_text(json.dumps({p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
    archive=shutil.make_archive(str(ROOT/'dist'/out.name),'zip',root_dir=out.parent,base_dir=out.name)
    digest=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    Path(archive+'.sha256').write_text(digest+'  '+Path(archive).name+'\n',encoding='ascii')
    print(archive,flush=True);print('SHA256',digest,flush=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform',choices=['windows','linux'],default='windows')
    args=parser.parse_args()
    dist=(ROOT/'dist').resolve();dist.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.package-',dir=dist) as temp:
        staging=Path(temp).resolve();staging.relative_to(dist)
        build(staging,args.platform)
if __name__=='__main__':main()
