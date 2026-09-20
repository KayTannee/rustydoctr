param([int]$Seconds = 300, [switch]$Vram4GB)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
& "$repo/.venv-baseline/Scripts/python.exe" scripts/prepare_throughput.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$profileName = if ($Vram4GB) { '4gb' } else { 'auto' }
$resultDir = Join-Path $repo "pybaseline/results/throughput/watch-$profileName-$stamp"
Write-Host "Results: $resultDir"
if ($Vram4GB) {
    & "$PSScriptRoot/run_pipeline.ps1" --vram-4gb --seconds $Seconds --output $resultDir
} else {
    & "$PSScriptRoot/run_pipeline.ps1" --auto --seconds $Seconds --output $resultDir
}
exit $LASTEXITCODE
