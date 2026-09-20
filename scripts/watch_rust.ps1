param([ValidateRange(1,86400)][int]$Seconds = 300)
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$suffix = [guid]::NewGuid().ToString('N').Substring(0,8)
$output = "pybaseline/results/manual_rust/$stamp-$suffix"
Write-Host 'Rust upright word-only slice: DB ResNet34 + PARSeq, 1024, recognition batch 128.'
Write-Host 'Model loading / CUDA algorithm warmup precede measurement and may take a minute.'
Write-Host 'This is an in-memory page test, not the Python feeder/writer streaming benchmark.'
& "$repo/scripts/run_rust.ps1" --image testdata/generated/a4_control.png --output $output --seconds $Seconds
exit $LASTEXITCODE
