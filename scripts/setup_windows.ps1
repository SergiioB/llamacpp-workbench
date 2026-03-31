#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-click setup script for llama-webui on Windows
.DESCRIPTION
    This script sets up the complete llama-webui environment on Windows including:
    - Python virtual environment
    - Python dependencies
    - llama.cpp cloning
    - Optional: CUDA build
.PARAMETER SkipBuild
    Skip building llama.cpp (just set up Python environment)
.PARAMETER UsePrebuilt
    Download pre-built binaries instead of building from source
.PARAMETER CudaVersion
    CUDA version to use for pre-built binaries (default: 12.4)
.EXAMPLE
    .\setup_windows.ps1
.EXAMPLE
    .\setup_windows.ps1 -UsePrebuilt -CudaVersion 12.8
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$UsePrebuilt,
    [string]$CudaVersion = "12.4"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$LlamaDir = Join-Path $RepoRoot "third_party" "llama.cpp"

Write-Host "=== llama-webui Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python
$python = Get-Command "python3" -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command "python" -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Error "Python not found. Please install Python 3.11 or later from https://python.org"
    exit 1
}

$pythonVersion = & $python.Source --version
Write-Host "Python found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
$VenvDir = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    & $python.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists at: $VenvDir" -ForegroundColor Green
}

# Activate virtual environment
$VenvScripts = Join-Path $VenvDir "Scripts"
$env:PATH = "$VenvScripts;$env:PATH"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& (Join-Path $VenvScripts "python.exe") -m pip install --upgrade pip

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
& (Join-Path $VenvScripts "pip.exe") install -e $RepoRoot
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Python dependencies"
    exit 1
}

Write-Host "Python dependencies installed successfully!" -ForegroundColor Green
Write-Host ""

# Clone llama.cpp
if (-not (Test-Path $LlamaDir)) {
    Write-Host "Cloning llama.cpp..." -ForegroundColor Yellow
    $ThirdPartyDir = Join-Path $RepoRoot "third_party"
    if (-not (Test-Path $ThirdPartyDir)) {
        New-Item -ItemType Directory -Path $ThirdPartyDir | Out-Null
    }
    git clone https://github.com/ggerganov/llama.cpp.git $LlamaDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to clone llama.cpp"
        exit 1
    }
} else {
    Write-Host "llama.cpp already cloned at: $LlamaDir" -ForegroundColor Green
}

if ($SkipBuild) {
    Write-Host ""
    Write-Host "=== Setup Complete (Build Skipped) ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "To build llama.cpp later, run:" -ForegroundColor Yellow
    Write-Host "  .\scripts\build_llama_cuda.ps1" -ForegroundColor Gray
    exit 0
}

# Download pre-built binaries or build from source
if ($UsePrebuilt) {
    Write-Host ""
    Write-Host "Downloading pre-built binaries..." -ForegroundColor Yellow
    
    $ReleaseVersion = "b8599"  # Adjust as needed
    $LlamaBinUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$ReleaseVersion/llama-$ReleaseVersion-bin-win-cuda-$CudaVersion-x64.zip"
    $CudartUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$ReleaseVersion/cudart-llama-bin-win-cuda-$CudaVersion-x64.zip"
    
    $PrebuiltDir = Join-Path $LlamaDir "prebuilt"
    if (-not (Test-Path $PrebuiltDir)) {
        New-Item -ItemType Directory -Path $PrebuiltDir | Out-Null
    }
    
    $LlamaZip = Join-Path $PrebuiltDir "llama-bin.zip"
    $CudartZip = Join-Path $PrebuiltDir "cudart.zip"
    
    Write-Host "Downloading llama.cpp binaries..." -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $LlamaBinUrl -OutFile $LlamaZip -UseBasicParsing
        Write-Host "Downloading CUDA runtime DLLs..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $CudartUrl -OutFile $CudartZip -UseBasicParsing
    } catch {
        Write-Error "Failed to download pre-built binaries. Try building from source instead."
        exit 1
    }
    
    Write-Host "Extracting binaries..." -ForegroundColor Gray
    Expand-Archive -Path $LlamaZip -DestinationPath $PrebuiltDir -Force
    Expand-Archive -Path $CudartZip -DestinationPath $PrebuiltDir -Force
    
    $ServerExe = Get-ChildItem -Path $PrebuiltDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
    if (-not $ServerExe) {
        Write-Error "llama-server.exe not found in extracted archive"
        exit 1
    }
    
    Write-Host ""
    Write-Host "=== Setup Complete (Pre-built) ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "llama-server location: $($ServerExe.FullName)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To use with llama-webui, set:" -ForegroundColor Yellow
    Write-Host "  `$env:LLAMA_WEBUI_LLAMA_SERVER = '$($ServerExe.FullName)'" -ForegroundColor Gray
    
} else {
    # Build from source
    Write-Host ""
    Write-Host "Building llama.cpp from source..." -ForegroundColor Yellow
    Write-Host "This requires CMake, Visual Studio Build Tools, and CUDA toolkit."
    Write-Host ""
    
    $BuildScript = Join-Path $ScriptDir "build_llama_cuda.ps1"
    if (Test-Path $BuildScript) {
        & $BuildScript
    } else {
        Write-Error "Build script not found: $BuildScript"
        exit 1
    }
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment:" -ForegroundColor Yellow
Write-Host "  $VenvScripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "To run llama-webui:" -ForegroundColor Yellow
Write-Host "  llama-webui" -ForegroundColor Gray
Write-Host ""
Write-Host "Open your browser to: http://localhost:8095" -ForegroundColor Cyan
