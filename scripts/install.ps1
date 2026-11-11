<#
.SYNOPSIS
  Installs Qwen RSI on Windows 11: Python, Ollama, the qwen3.8:27b model,
  a private virtual environment and a Start menu shortcut.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -SkipModel
#>
[CmdletBinding()]
param(
    [string]$Model = "qwen3.8:27b",
    [switch]$SkipModel,
    [switch]$NoShortcut
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $env:LOCALAPPDATA "QwenRSI"
$Venv = Join-Path $AppDir "venv"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --- OS check --------------------------------------------------------------
$build = [System.Environment]::OSVersion.Version.Build
if ($build -lt 22000) { Write-Warning "Windows build $build detected; this app targets Windows 11 (build 22000+)." }

# --- Python ----------------------------------------------------------------
Step "Checking Python 3.11+"
function Find-Python {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3.13") }, @{ Exe = "py"; Args = @("-3.12") },
        @{ Exe = "py"; Args = @("-3.11") }, @{ Exe = "python"; Args = @() })
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        $a = $c.Args
        try {
            $ver = & $c.Exe @a -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver -and ([version]"$ver".Trim() -ge [version]"3.11")) { return $c }
        } catch { }
    }
    return $null
}
$py = Find-Python
if (-not $py) {
    Write-Host "Installing Python 3.12 with winget..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $py = Find-Python
    if (-not $py) { throw "Python 3.11+ still not found; open a new terminal and re-run this script." }
}
$pyArgs = $py.Args
Write-Host "Using: $($py.Exe) $($pyArgs -join ' ')"

# --- Ollama ----------------------------------------------------------------
Step "Checking Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Ollama with winget..."
    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null }
catch {
    Write-Host "Starting the Ollama server..."
    Start-Process -WindowStyle Hidden ollama -ArgumentList "serve"
    Start-Sleep -Seconds 5
}

# --- GPU memory hint ---------------------------------------------------------
Step "Checking GPU memory"
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    $mb = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    Write-Host "GPU memory: $mb MiB"
    if ([int]$mb -lt 20000) {
        Write-Warning "qwen3.8:27b (Q4_K_M, ~17 GB) will partly run on the CPU with less than ~20 GB VRAM. Expect it to be slow."
    }
} else {
    Write-Host "nvidia-smi not found; skipping VRAM check (AMD/Intel GPUs work with Ollama too)."
}

# --- Model -------------------------------------------------------------------
if (-not $SkipModel) {
    Step "Pulling $Model (about 18 GB, once)"
    ollama pull $Model
}

# --- Virtual environment + app ----------------------------------------------
Step "Creating virtual environment in $Venv"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
& $py.Exe @pyArgs -m venv $Venv
$vpy = Join-Path $Venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install "$Repo"

# --- Shortcut ----------------------------------------------------------------
if (-not $NoShortcut) {
    Step "Creating Start menu shortcut"
    $target = Join-Path $Venv "Scripts\qwen-rsi.exe"
    $lnk = Join-Path ([Environment]::GetFolderPath("Programs")) "Qwen RSI.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = $AppDir
    $sc.Description = "Recursive self-improvement on Qwen3.8 via Ollama"
    $sc.Save()
    Write-Host "Shortcut: $lnk"
}

Step "Self-check"
& (Join-Path $Venv "Scripts\qwen-rsi-cli.exe") check
Write-Host "`nDone. Start 'Qwen RSI' from the Start menu, or run: $(Join-Path $Venv 'Scripts\qwen-rsi.exe')" -ForegroundColor Green
