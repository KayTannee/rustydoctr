param([string]$PythonCommand = "py", [string]$PythonVersion = "3.12")
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path .venv/Scripts/python.exe)) {
    if ($PythonCommand -eq "py") {
        & py "-$PythonVersion" -m venv .venv
    } else {
        & $PythonCommand -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 x64 is required. Install it or create .venv with your chosen supported Python.' }
}
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $python verify_bundle.py
if ($LASTEXITCODE -ne 0) { throw 'Bundle checksum verification failed' }
$wheel = Get-Item (Join-Path $PSScriptRoot 'rustydoctr-0.2.0-cp312-abi3-win_amd64.whl')
& $python -m pip install -c runtime-lock.txt "$($wheel.FullName)[gpu,examples]"
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $python -m rustydoctr doctor --models models --config configs/low-vram.json --smoke
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke check failed; inspect the diagnostic output above' }
