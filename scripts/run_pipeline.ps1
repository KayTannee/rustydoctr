param([Parameter(ValueFromRemainingArguments=$true)][string[]]$OcrArgs)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$runtime = Join-Path $repo '.venv-baseline/Lib/site-packages/onnxruntime/capi'
$torchLib = Join-Path $repo '.venv-baseline/Lib/site-packages/torch/lib'
$env:ORT_DYLIB_PATH = Join-Path $runtime 'onnxruntime.dll'
$env:PATH = "$runtime;$torchLib;$env:PATH"
& "$repo/target/release/throughput.exe" @OcrArgs
exit $LASTEXITCODE
