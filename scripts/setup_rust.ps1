$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
if (-not (Test-Path '.venv-baseline/Scripts/python.exe')) { throw 'Run pybaseline/setup.ps1 first.' }
uv pip install --python .venv-baseline/Scripts/python.exe --cache-dir .cache/uv -r pybaseline/requirements-rust.txt
if ($LASTEXITCODE) { throw 'ONNX dependencies failed' }
& .venv-baseline/Scripts/python.exe scripts/export_rust_models.py
if ($LASTEXITCODE) { throw 'Model export failed' }
$cargoCommand = Get-Command cargo -ErrorAction SilentlyContinue
$cargo = if ($cargoCommand) { $cargoCommand.Source } else { Join-Path $env:USERPROFILE '.cargo/bin/cargo.exe' }
$env:CARGO_HOME = Join-Path $repo '.cache/cargo'
& $cargo build --release --locked
if ($LASTEXITCODE) { throw 'Rust build failed' }
