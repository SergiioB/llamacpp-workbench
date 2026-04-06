# Windows Setup Guide for llama-webui with RTX 5060

This guide documents the complete setup process for running `llama-webui` on Windows with an NVIDIA RTX 5060 Laptop GPU (dual GPU system with Intel Arc iGPU + NVIDIA dGPU).

## Hardware Configuration

- **CPU**: Intel Core Ultra 9 285H
- **iGPU**: Intel Arc 140T GPU (16GB) - Used for power saving
- **dGPU**: NVIDIA GeForce RTX 5060 Laptop GPU - For CUDA acceleration
- **Architecture**: Blackwell (compute capability 12.0)
- **RAM**: ~32 GB

## Prerequisites

### Required Software

The following components must be installed before building or running llama.cpp with GPU support:

#### 1. NVIDIA GPU Driver
**Status**: Driver files exist but GPU shows "Unknown" status

**Issue**: In dual GPU laptops, the NVIDIA GPU may be in power-saving mode

**Resolution**:
```powershell
# Check GPU status
Get-PnpDevice -Class Display | Select-Object Name, Status

# If NVIDIA GPU shows "Unknown", you may need to:
# 1. Open NVIDIA Control Panel
# 2. Set preferred GPU to "High-performance NVIDIA processor"
# 3. Or disable Intel GPU in BIOS if mux switch is available
```

**Download**: https://www.nvidia.com/drivers

#### 2. CUDA Toolkit
**Status**: NOT INSTALLED
**Path Checked**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`

**Required Version**: CUDA 12.8 or later (for RTX 5060 Blackwell architecture)

**Download**: https://developer.nvidia.com/cuda-downloads

**Verification**:
```powershell
nvcc --version
```

#### 3. CMake
**Status**: NOT INSTALLED
**Required Version**: 3.18 or later

**Installation Options**:
```powershell
# Option 1: Using winget
winget install Kitware.CMake

# Option 2: Download from https://cmake.org/download/
```

**Verification**:
```powershell
cmake --version
```

#### 4. Visual Studio 2022 Build Tools
**Status**: NOT INSTALLED

**Required Components**:
- MSVC v143 - VS 2022 C++ x64/x86 build tools
- Windows 10/11 SDK
- C++ CMake tools for Windows

**Installation**:
```powershell
# Using winget
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"
```

Or download from: https://aka.ms/vs/17/release/vs_BuildTools.exe

Select workload: **"Desktop development with C++"**

**Verification**:
```powershell
cl
```

## Setup Instructions

### Step 1: Clone the Repository

```powershell
git clone https://github.com/SergiioB/llamacpp-workbench.git
cd llamacpp-workbench
```

### Step 2: Create Python Virtual Environment

```powershell
python3 -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```powershell
pip install --upgrade pip
pip install -e .
```

This installs:
- `fastapi>=0.115.0`
- `uvicorn>=0.30.0`

### Step 4: Clone llama.cpp

```powershell
git clone https://github.com/ggerganov/llama.cpp.git third_party\llama.cpp
```

### Step 5: Build llama.cpp with CUDA Support

**Prerequisites**: Ensure CUDA, CMake, and MSVC are installed

```powershell
cd third_party\llama.cpp
mkdir build-cuda
cd build-cuda

cmake .. `
    -DGGML_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_CLI=ON

cmake --build . --config Release -j
```

**Expected Output**:
- `llama-server.exe` in `build-cuda\bin\Release\` or `build-cuda\bin\`
- `llama-cli.exe` in same directory

### Alternative: Use Pre-built Binaries

If building from source fails, download pre-built binaries:

```powershell
# Download from GitHub releases
curl.exe -L -o llama-bin.zip https://github.com/ggml-org/llama.cpp/releases/download/b8599/llama-b8599-bin-win-cuda-12.4-x64.zip

# Also download CUDA runtime DLLs if needed
curl.exe -L -o cudart.zip https://github.com/ggml-org/llama.cpp/releases/download/b8599/cudart-llama-bin-win-cuda-12.4-x64.zip

# Extract
Expand-Archive llama-bin.zip -DestinationPath .\third_party\llama.cpp\prebuilt
Expand-Archive cudart.zip -DestinationPath .\third_party\llama.cpp\prebuilt
```

### Step 6: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Windows paths use double backslashes or forward slashes
LLAMA_WEBUI_LLAMA_SERVER=C:\Users\<username>\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-server.exe
LLAMA_WEBUI_LLAMA_CLI=C:\Users\<username>\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-cli.exe

# Optional: Model directories
LLAMA_WEBUI_MODEL_DIRS=C:\Users\<username>\models;C:\Users\<username>\llamacpp-workbench\models
LLAMA_WEBUI_DATA_DIR=C:\Users\<username>\llamacpp-workbench\data
```

Or set via PowerShell:
```powershell
$env:LLAMA_WEBUI_LLAMA_SERVER = "C:\Users\$env:USERNAME\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-server.exe"
$env:LLAMA_WEBUI_LLAMA_CLI = "C:\Users\$env:USERNAME\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-cli.exe"
```

### Step 7: Download a Model

Download a GGUF model from HuggingFace:

**Recommended for RTX 5060 (8GB VRAM)**:
- `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~4.5GB)
- `Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.9GB)

**Recommended for testing (small, fast)**:
- `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1GB)

```powershell
# Create models directory
mkdir models

# Download using curl or browser
curl.exe -L -o models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf `
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Step 8: Run the Application

```powershell
# Activate venv if not already active
.venv\Scripts\activate

# Run the web UI
llama-webui
```

Open browser to: http://localhost:8095

## GPU Configuration for Dual GPU Laptops

### Switching to NVIDIA GPU

On laptops with dual GPUs, Windows may use the Intel iGPU by default. To force NVIDIA:

#### Method 1: NVIDIA Control Panel
1. Open NVIDIA Control Panel
2. Go to "Manage 3D settings"
3. Select "Program Settings" tab
4. Add `llama-server.exe` and `python.exe`
5. Set preferred graphics processor to "High-performance NVIDIA processor"

#### Method 2: Windows Graphics Settings
1. Open Windows Settings → System → Display → Graphics
2. Add `llama-server.exe` as a custom app
3. Set to "High performance"

#### Method 3: Environment Variable
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
```

### Verify GPU is Being Used

```powershell
# While running llama-webui, check in another terminal
nvidia-smi
```

You should see `llama-server.exe` in the process list with GPU memory usage.

## Troubleshooting

### Issue: "nvcc not found"
**Cause**: CUDA toolkit not installed or not in PATH
**Fix**: Add CUDA to PATH:
```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
```

### Issue: "cl not found" during build
**Cause**: MSVC not installed or not in PATH
**Fix**: Run from Developer Command Prompt or use:
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### Issue: "CUDA architecture not supported"
**Cause**: RTX 5060 uses Blackwell (sm_100) which requires CUDA 12.8+
**Fix**: Update CUDA toolkit to 12.8 or later

### Issue: "Unknown" status for NVIDIA GPU
**Cause**: GPU in power-saving mode or driver issue
**Fix**:
1. Update NVIDIA drivers
2. Disable Intel GPU in BIOS (if mux switch available)
3. Force NVIDIA GPU via control panel

### Issue: Out of memory errors
**Fix**: Reduce GPU layers or use quantization:
```json
{
  "gpu_layers": 20,
  "ctx_size": 4096
}
```

## Current Status Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| Python venv | ✅ | Created at `.venv` |
| Python deps | ✅ | fastapi, uvicorn installed |
| llama.cpp source | ✅ | Cloned to `third_party/llama.cpp` |
| CMake | ❌ | Not installed |
| MSVC Build Tools | ❌ | Not installed |
| CUDA Toolkit | ❌ | Not installed |
| llama-server built | ❌ | Requires above tools |
| Model downloaded | ❌ | Pending |
| GPU working | ❌ | Shows "Unknown" status |

## Quick Reference Commands

```powershell
# Check GPU status
Get-PnpDevice -Class Display | Select-Object Name, Status

# Check NVIDIA GPU info (admin required)
nvidia-smi

# Check if tools are installed
Get-Command cmake, cl, nvcc

# Activate venv
.venv\Scripts\activate

# Run web UI
llama-webui
```

## Additional Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama.cpp Releases](https://github.com/ggerganov/llama.cpp/releases)
- [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
- [CMake Downloads](https://cmake.org/download/)

## Notes for PR

This documentation was created to help Windows users with RTX 5060 laptops set up llama-webui. Key additions needed:

1. Windows-specific setup instructions
2. Dual GPU laptop configuration
3. Pre-built binary download links
4. Environment variable configuration for Windows paths
5. Troubleshooting section for common Windows issues

---

*Last updated: 2026-03-31*
