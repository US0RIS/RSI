<#
.SYNOPSIS
  Creates the training environment used by the RSI button (PyTorch + CUDA + Unsloth)
  in %LOCALAPPDATA%\QwenRSI\trainer-venv and verifies the GPU.

.NOTES
  Needs an NVIDIA GPU with ~24 GB VRAM for QLoRA on the 27B model, a recent NVIDIA driver,
  and about 40 GB of free disk (the training copy of the weights is downloaded from
  Hugging Face on the first round; the merged-weights fallback needs ~60 GB more).
  If the check fails on native Windows, create the environment in WSL2 instead and point
  Settings > "Trainer Python" at a small .cmd wrapper that calls it.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_trainer.ps1
#>
[CmdletBinding()]
param(
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $env:LOCALAPPDATA "QwenRSI"
$Venv = Join-Path $AppDir "trainer-venv"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Checking NVIDIA driver"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi not found. Install the current NVIDIA driver first (training needs CUDA)."
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

Step "Finding Python 3.12 or 3.11 (PyTorch wheels lag behind the newest Python)"
$py = $null
foreach ($v in @("-3.12", "-3.11")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py $v -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $v; break }
    }
}
if (-not $py) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    $py = "-3.12"
}

Step "Creating $Venv"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
& py $py -m venv $Venv
$vpy = Join-Path $Venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip

Step "Installing PyTorch with CUDA ($TorchIndex)"
& $vpy -m pip install torch --index-url $TorchIndex

Step "Installing Unsloth"
& $vpy -m pip install unsloth

Step "Verifying"
& $vpy (Join-Path $Repo "qwen_rsi\selftrain\train_lora.py") check
if ($LASTEXITCODE -ne 0) {
    Write-Warning "The trainer check failed (see the QRSI line above). The RSI button will still evolve the agent and collect verified training data, but cannot train weights until this passes."
} else {
    Write-Host "`nTrainer ready. The RSI button will use $vpy" -ForegroundColor Green
}
