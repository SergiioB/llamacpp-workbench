# llama.cpp Setup - Missing Components Troubleshooting

This guide helps identify and resolve missing components needed to build and run llama.cpp with GPU support on Windows.

## Checking Your System

Run these commands to determine what is installed and what is missing:

```powershell
# Check NVIDIA GPU
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Check CMake
cmake --version

# Check MSVC (Visual Studio)
cl

# Check all at once
Get-Command cmake, cl, nvcc -ErrorAction SilentlyContinue
```

## Common Missing Components

### 1. NVIDIA Driver Issues

**Symptoms**: `nvidia-smi` fails, or GPU shows "Unknown" in Device Manager.

**Diagnosis**:
```powershell
Get-PnpDevice -Class Display | Select-Object Name, Status
```

**Solutions**:
- Install or update drivers from https://www.nvidia.com/drivers
- On dual-GPU laptops, the NVIDIA GPU may be in power-saving mode
- Try disabling the integrated GPU in BIOS if a mux switch is available
- Configure NVIDIA Control Panel to use the high-performance GPU

### 2. CUDA Toolkit Not Installed

**Symptoms**: `nvcc` command not found, or `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` does not exist.

**Required Version**: CUDA 12.4 or later. CUDA 12.8+ is needed for RTX 50-series GPUs.

**Installation**: Download from https://developer.nvidia.com/cuda-downloads

**Verification**:
```powershell
nvcc --version
# Should show CUDA version 12.4 or later
```

### 3. CMake Not Installed

**Symptoms**: `cmake` command not found.

**Required Version**: 3.18 or later.

**Installation**:
```powershell
winget install Kitware.CMake
```

Or download from: https://cmake.org/download/

### 4. Visual Studio / MSVC Build Tools Not Installed

**Symptoms**: `cl` command not found.

**Required**: Visual Studio 2022 with C++ build tools.

**Components Needed**:
- MSVC v143 - VS 2022 C++ x64/x86 build tools
- Windows 10/11 SDK
- C++ CMake tools for Windows

**Installation**:
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"
```

Or download from: https://visualstudio.microsoft.com/downloads/

Select the **"Desktop development with C++"** workload.

## Quick Setup Steps

If multiple components are missing, install them in this order:

1. **NVIDIA drivers** (if GPU not detected)
2. **Visual Studio 2022 Build Tools** (includes MSVC)
3. **CMake**
4. **CUDA Toolkit** (must match your GPU's compute capability)

Then verify everything:
```powershell
nvidia-smi          # GPU driver OK
cl                  # MSVC OK
cmake --version     # CMake OK
nvcc --version      # CUDA OK
```

## Building llama.cpp After Setup

```powershell
cd <project-root>\third_party\llama.cpp
mkdir build-cuda
cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_CLI=ON
cmake --build . --config Release -j
```

Or use the automated setup:
```powershell
.\scripts\setup_windows.ps1
```

## Configuring llama-webui

After building, configure the binary path:

```powershell
# Option 1: Use the configure script
.\scripts\configure_env.ps1 -CreateEnvFile

# Option 2: Set manually
$env:LLAMA_WEBUI_LLAMA_SERVER = "<project-root>\third_party\llama.cpp\build-cuda\bin\Release\llama-server.exe"
```

Replace `<project-root>` with the actual path to your cloned repository.

## Alternative: Pre-built Binaries

If building from source is problematic, download pre-built binaries from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases). Choose the file matching your CUDA version and architecture (e.g., `llama-bXXXX-bin-win-cuda-cu12.4-x64.zip`).

## Verifying Everything Works

```powershell
# Activate venv
.venv\Scripts\activate

# Run llama-webui
llama-webui
```

Open http://localhost:8095 in your browser.
