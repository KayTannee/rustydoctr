param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$env:UV_CACHE_DIR = "$repo/.cache/uv"
$env:CARGO_HOME = "$repo/.cache/cargo"
$env:PATH = "$env:USERPROFILE/.cargo/bin;$env:PATH"
$python = "$repo/.venv-baseline/Scripts/python.exe"
uv pip install --python $python 'maturin==1.15.0'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m maturin build --release --locked --out dist
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$wheel = Get-ChildItem "$repo/dist/rustydoctr-*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
uv pip install --python $python --reinstall --no-deps $wheel.FullName
exit $LASTEXITCODE
