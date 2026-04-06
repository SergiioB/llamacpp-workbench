# llama.cpp Setup - Missing Components Documentation

## Hardware Detected
- **CPU**: Intel Core Ultra 9 285H
- **iGPU**: Intel Arc 140T GPU (16GB) - Active
- **dGPU**: NVIDIA GeForce RTX 5060 Laptop GPU - Present but shows "Unknown" status

## Current Status
The RTX 5060 Laptop GPU exists in the system but is not properly initialized (shows "Unknown" status in PnpDevice).

## Missing Components for GPU Support

### 1. NVIDIA Driver Issues
**Problem**: RTX 5060 shows "Unknown" status in device manager
**Evidence**:
```powershell
Get-PnpDevice -Class Display | Select-Object Name, Status
# Output: NVIDIA GeForce RTX 5060 Laptop GPU Unknown
```

**Solution Required**:
- Run as Administrator and check device status
- May need to disable Intel GPU in BIOS (mux switch) or enable NVIDIA GPU
- Install latest NVIDIA drivers from https://www.nvidia.com/drivers

### 2. CUDA Toolkit
**Status**: NOT INSTALLED
**Check**:
```powershell
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
# Result: False
```

**Required Version**: CUDA 12.8 or later (for RTX 5060 Blackwell architecture)
**Download**: https://developer.nvidia.com/cuda-downloads

### 3. CMake
**Status**: NOT INSTALLED
**Required**: CMake 3.18 or later
**Download**: https://cmake.org/download/

### 4. Visual Studio / MSVC Build Tools
**Status**: NOT INSTALLED
**Required**: Visual Studio 2022 with C++ build tools
**Components Needed**:
- MSVC v143 - VS 2022 C++ x64/x86 build tools
- Windows 10/11 SDK
- C++ CMake tools for Windows

**Download**: https://visualstudio.microsoft.com/downloads/
(Select "Desktop development with C++" workload)

### 5. Python Development Tools (Optional but recommended)
The Python dependencies are installed, but for building native extensions:
- Visual C++ 14.0 or greater required

## Setup Steps Required

### Step 1: Fix NVIDIA GPU Detection
1. Open Device Manager as Administrator
2. Check NVIDIA GPU status under Display adapters
3. If showing error code, try:
   - Update driver from NVIDIA website
   - Disable iGPU in BIOS if mux switch available
   - Or configure NVIDIA Control Panel for preferred GPU

### Step 2: Install Visual Studio 2022 Build Tools
```powershell
# Using winget (if available):
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"
```

Or download from: https://aka.ms/vs/17/release/vs_BuildTools.exe

### Step 3: Install CMake
```powershell
# Using winget:
winget install Kitware.CMake

# Or download installer from cmake.org
```

### Step 4: Install CUDA Toolkit
Download from: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64

For RTX 5060 (Blackwell architecture), use CUDA 12.8 or later.

### Step 5: Verify Installation
```powershell
# Verify CUDA:
nvcc --version

# Verify CMake:
cmake --version

# Verify MSVC:
cl

# Verify NVIDIA GPU:
nvidia-smi
```

### Step 6: Build llama.cpp with CUDA
```powershell
cd C:\Users\sergi\llamacpp-workbench\third_party\llama.cpp
mkdir build-cuda
cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_CLI=ON
cmake --build . --config Release -j
```

### Step 7: Configure llama-webui
Set environment variable:
```powershell
$env:LLAMA_WEBUI_LLAMA_SERVER = "C:\Users\sergi\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\llama-server.exe"
```

Or create a `.env` file in the project root with:
```
LLAMA_WEBUI_LLAMA_SERVER=C:\Users\sergi\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\llama-server.exe
```

## Alternative: Pre-built Binaries
If building is problematic, you can download pre-built llama.cpp binaries with CUDA support:
- https://github.com/ggerganov/llama.cpp/releases

Look for `llama-bXXXX-bin-win-cuda-cu12.4-x64.zip` (or cu12.8 for RTX 5060)

## Current Working State
- Python virtual environment: ✅ Created at `.venv`
- Python dependencies: ✅ Installed (fastapi, uvicorn)
- llama.cpp source: ✅ Cloned to `third_party/llama.cpp`
- llama-server: ❌ Not built (requires CMake + MSVC + CUDA)
- GPU acceleration: ❌ Not available (CUDA not installed)

## Quick Test After Setup
```powershell
# Activate venv
.venv\Scripts\activate

# Download a small model for testing
# e.g., Qwen2.5-1.5B-Instruct-GGUF from HuggingFace

# Run llama-webui
llama-webui
```

The web UI will be available at http://localhost:8095
