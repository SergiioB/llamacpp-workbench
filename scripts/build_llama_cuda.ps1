#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build llama.cpp with CUDA support for NVIDIA GPUs on Windows
.DESCRIPTION
    This script clones (if needed) and builds llama.cpp with CUDA support.
    It automatically detects CUDA installation and configures the build.
.PARAMETER Clean
    Clean build directory before building
.PARAMETER Jobs
    Number of parallel build jobs (default: number of processors)
.EXAMPLE
    .\build_llama_cuda.ps1
.EXAMPLE
    .\build_llama_cuda.ps1 -Clean -Jobs 8
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [int]$Jobs = $env:NUMBER_OF_PROCESSORS
)

$ErrorActionPreference = "Stop"

# Determine script location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$LlamaDir = Join-Path $RepoRoot "third_party" "llama.cpp"
$BuildDir = Join-Path $LlamaDir "build-cuda"

Write-Host "=== llama.cpp CUDA Build Script for Windows ===" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
function Test-Prerequisite {
    param([string]$Name, [string]$Command)
    $cmd = Get-Command $Command -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Error "Prerequisite not found: $Name ($Command). Please install it first."
        return $false
    }
    Write-Host "  $Name`: OK" -ForegroundColor Green
    return $true
}

Write-Host "Checking prerequisites..." -ForegroundColor Yellow
$prereqsOk = $true
$prereqsOk = (Test-Prerequisite "Git" "git") -and $prereqsOk
$prereqsOk = (Test-Prerequisite "CMake" "cmake") -and $prereqsOk

# Check for CUDA
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    if (-not (Test-Path $cudaPath)) {
        $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    }
    if (-not (Test-Path $cudaPath)) {
        $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    }
}

if (Test-Path $cudaPath) {
    Write-Host "  CUDA: OK ($cudaPath)" -ForegroundColor Green
    $env:PATH = "$cudaPath\bin;$env:PATH"
} else {
    Write-Error "CUDA not found. Please install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
    exit 1
}

# Check for MSVC
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsPath) {
        Write-Host "  Visual Studio: OK ($vsPath)" -ForegroundColor Green
        Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
        Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64"
    } else {
        Write-Error "Visual Studio C++ tools not found. Please install 'Desktop development with C++' workload."
        exit 1
    }
} else {
    # Try to find cl.exe in PATH
    $cl = Get-Command "cl" -ErrorAction SilentlyContinue
    if (-not $cl) {
        Write-Error "Visual Studio not found. Please install Visual Studio 2022 Build Tools."
        exit 1
    }
    Write-Host "  Visual Studio: OK (found in PATH)" -ForegroundColor Green
}

if (-not $prereqsOk) {
    exit 1
}

Write-Host ""

# Clone llama.cpp if needed
if (-not (Test-Path $LlamaDir)) {
    Write-Host "Cloning llama.cpp..." -ForegroundColor Yellow
    $ThirdPartyDir = Join-Path $RepoRoot "third_party"
    if (-not (Test-Path $ThirdPartyDir)) {
        New-Item -ItemType Directory -Path $ThirdPartyDir | Out-Null
    }
    git clone https://github.com/ggerganov/llama.git (Join-Path $ThirdPartyDir "llama.cpp")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to clone llama.cpp"
        exit 1
    }
} else {
    Write-Host "llama.cpp already cloned at: $LlamaDir" -ForegroundColor Green
}

# Clean build directory if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

Write-Host ""
Write-Host "Configuring build with CUDA support..." -ForegroundColor Yellow
Write-Host "Build directory: $BuildDir"

# Configure with CMake
$cmakeArgs = @(
    "..",
    "-DGGML_CUDA=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_CLI=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=native"
)

Push-Location $BuildDir
try {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed"
        exit 1
    }

    Write-Host ""
    Write-Host "Building llama.cpp with $Jobs parallel jobs..." -ForegroundColor Yellow
    & cmake --build . --config Release -j $Jobs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed"
        exit 1
    }
} finally {
    Pop-Location
}

# Determine output directory
$BinDir = Join-Path $BuildDir "bin"
if (-not (Test-Path $BinDir)) {
    $BinDir = Join-Path $BuildDir "bin\Release"
}

$ServerExe = Join-Path $BinDir "llama-server.exe"
$CliExe = Join-Path $BinDir "llama-cli.exe"

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Binaries location: $BinDir" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $ServerExe) {
    Write-Host "  llama-server.exe: $ServerExe" -ForegroundColor Green
} else {
    Write-Warning "  llama-server.exe not found in expected location"
}

if (Test-Path $CliExe) {
    Write-Host "  llama-cli.exe: $CliExe" -ForegroundColor Green
} else {
    Write-Warning "  llama-cli.exe not found in expected location"
}

Write-Host ""
Write-Host "To use with llama-webui:" -ForegroundColor Yellow
Write-Host "  1. Set environment variable:" -ForegroundColor White
Write-Host "     `$env:LLAMA_WEBUI_LLAMA_SERVER = '$ServerExe'" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Or create a .env file in the project root with:" -ForegroundColor White
Write-Host "     LLAMA_WEBUI_LLAMA_SERVER=$ServerExe" -ForegroundColor Gray
Write-Host ""
