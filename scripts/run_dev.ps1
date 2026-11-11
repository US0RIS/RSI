# Run from a source checkout without installing (development).
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
python -m qwen_rsi @args
