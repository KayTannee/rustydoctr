param([string]$Python = '3.12')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv-baseline/Scripts/python.exe')) {
    uv venv .venv-baseline --python $Python --cache-dir .cache/uv
    if ($LASTEXITCODE) { throw 'Failed to create environment' }
}
uv pip install --python .venv-baseline/Scripts/python.exe --cache-dir .cache/uv 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'Failed to install CUDA PyTorch' }
$requirements = if (Test-Path "$PSScriptRoot/requirements-lock.txt") { "$PSScriptRoot/requirements-lock.txt" } else { "$PSScriptRoot/requirements.txt" }
uv pip install --python .venv-baseline/Scripts/python.exe --cache-dir .cache/uv -r $requirements
if ($LASTEXITCODE) { throw 'Failed to install dependencies' }
& .venv-baseline/Scripts/python.exe -c "import torch, doctr; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, doctr.__version__, torch.cuda.get_device_name()); print(torch.ones(1, device='cuda'))"
if ($LASTEXITCODE) { throw 'CUDA verification failed' }
