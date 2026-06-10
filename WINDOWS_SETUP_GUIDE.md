# Windows Setup Guide for llama-webui

This guide covers setting up `llama-webui` on Windows with NVIDIA GPU acceleration via CUDA.

## Hardware Requirements

- **GPU**: Any NVIDIA GPU with CUDA support (compute capability 6.0+)
- **VRAM**: 4 GB minimum; 8 GB+ recommended for 7B+ models
- **RAM**: 8 GB minimum; 16 GB+ recommended
- **CPU**: Any modern x86_64 processor

### Dual GPU Laptops

On laptops with both an integrated GPU (Intel/AMD) and an NVIDIA discrete GPU, Windows may default to the integrated GPU for power saving. You will need to force NVIDIA GPU usage (see the [Dual GPU Configuration](#gpu-configuration-for-dual-gpu-laptops) section below).

## Prerequisites

### Required Software

The following components must be installed before building or running llama.cpp with GPU support:

#### 1. NVIDIA GPU Driver

**Download**: https://www.nvidia.com/drivers

**Verification**:
```powershell
nvidia-smi
```

If the GPU shows as "Unknown" in Device Manager on dual-GPU laptops, see the troubleshooting section below.

#### 2. CUDA Toolkit

**Required Version**: CUDA 12.4 or later. Use CUDA 12.8+ for RTX 50-series (Blackwell) GPUs.

**Download**: https://developer.nvidia.com/cuda-downloads

**Verification**:
```powershell
nvcc --version
```

#### 3. CMake

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

### Step 4: Build llama.cpp with CUDA Support

**Option A: Use the setup script (recommended)**

```powershell
.\scripts\setup_windows.ps1
```

**Option B: Build from source manually**

```powershell
git clone https://github.com/ggerganov/llama.cpp.git third_party\llama.cpp
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

**Option C: Use Pre-built Binaries**

```powershell
.\scripts\setup_windows.ps1 -UsePrebuilt
```

Or download manually from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases). Choose the file matching your CUDA version (e.g., `llama-bXXXX-bin-win-cuda-cu12.4-x64.zip`).

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Replace <username> with your Windows username
LLAMA_WEBUI_LLAMA_SERVER=C:\Users\<username>\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-server.exe
LLAMA_WEBUI_LLAMA_CLI=C:\Users\<username>\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-cli.exe

# Optional: Model directories
LLAMA_WEBUI_MODEL_DIRS=C:\Users\<username>\models;C:\Users\<username>\llamacpp-workbench\models
LLAMA_WEBUI_DATA_DIR=C:\Users\<username>\llamacpp-workbench\data
```

Or use the configure script:
```powershell
.\scripts\configure_env.ps1 -CreateEnvFile
```

Or set via PowerShell:
```powershell
$env:LLAMA_WEBUI_LLAMA_SERVER = "C:\Users\$env:USERNAME\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-server.exe"
$env:LLAMA_WEBUI_LLAMA_CLI = "C:\Users\$env:USERNAME\llamacpp-workbench\third_party\llama.cpp\build-cuda\bin\Release\llama-cli.exe"
```

### Step 6: Download a Model

Download a GGUF model from HuggingFace:

**Recommended for GPUs with 6-8 GB VRAM**:
- `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~4.5GB)
- `Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.9GB)

**Recommended for GPUs with 4 GB VRAM**:
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf` (~2GB)

**Recommended for testing (small, fast)**:
- `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1GB)

```powershell
# Create models directory
mkdir models

# Download using curl or browser
curl.exe -L -o models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf `
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Step 7: Run the Application

```powershell
# Activate venv if not already active
.venv\Scripts\activate

# Run the web UI
llama-webui
```

Open browser to: http://localhost:8095

## Browser Inference (No Build Required)

If you just want to try LLM inference quickly without building llama.cpp, `llama-webui` supports running models directly in the browser via WebGPU.

**Requirements:**
- Google Chrome 113+ or Microsoft Edge 113+
- Any GPU (NVIDIA, AMD, Intel Arc)

**Steps:**

1. Start the server:
   ```powershell
   .venv\Scripts\activate
   llama-webui
   ```

2. Open http://localhost:8095 in Chrome or Edge

3. The sidebar shows **"Browser Inference (WebGPU)"** -- click **"Enable Browser Mode"**

4. Select a model (e.g., Qwen 2.5 1.5B) and click **"Load"**

5. Wait for the model to download (~1 GB for Qwen 1.5B, cached after first load)

6. Chat normally -- inference runs on your GPU via WebGPU

**Available browser models:**

| Model | Download Size | VRAM Needed |
|-------|--------------|-------------|
| SmolLM2 135M | ~100 MB | Any GPU |
| SmolLM2 360M | ~220 MB | Any GPU |
| Llama 3.2 1B | ~700 MB | 2 GB |
| Qwen 2.5 1.5B | ~1 GB | 2 GB |
| Qwen 2.5 3B | ~1.8 GB | 4 GB |
| Phi 3.5 Mini | ~2.3 GB | 4 GB |
| Gemma 2 2B | ~1.4 GB | 4 GB |
| Qwen 2.5 7B | ~4 GB | 8 GB |
| Llama 3.1 8B | ~4.5 GB | 8 GB |

**When to use browser vs server mode:**

- **Browser mode**: Quick testing, no-install, models up to ~8B parameters
- **Server mode** (with llama.cpp built): Larger models (27B+), fine-grained control, custom flags, higher throughput

## GPU Configuration for Dual GPU Laptops

On laptops with both integrated and discrete GPUs, Windows may use the integrated GPU by default. To force NVIDIA:

### Method 1: NVIDIA Control Panel
1. Open NVIDIA Control Panel
2. Go to "Manage 3D settings"
3. Select "Program Settings" tab
4. Add `llama-server.exe` and `python.exe`
5. Set preferred graphics processor to "High-performance NVIDIA processor"

### Method 2: Windows Graphics Settings
1. Open Windows Settings -> System -> Display -> Graphics
2. Add `llama-server.exe` as a custom app
3. Set to "High performance"

### Method 3: Environment Variable
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
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
```
Adjust the version number to match your installed CUDA version.

### Issue: "cl not found" during build
**Cause**: MSVC not installed or not in PATH
**Fix**: Run from Developer Command Prompt or use:
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### Issue: "CUDA architecture not supported"
**Cause**: CUDA toolkit version too old for your GPU architecture
**Fix**: Update CUDA toolkit. RTX 50-series (Blackwell) requires CUDA 12.8+. RTX 40/30/20-series generally works with CUDA 12.4+.

### Issue: "Unknown" status for NVIDIA GPU
**Cause**: GPU in power-saving mode or driver issue (common on dual-GPU laptops)
**Fix**:
1. Update NVIDIA drivers from https://www.nvidia.com/drivers
2. Disable integrated GPU in BIOS (if mux switch available)
3. Force NVIDIA GPU via NVIDIA Control Panel (see above)

### Issue: Out of memory errors
**Fix**: Reduce GPU layers or use a smaller model:
```json
{
  "gpu_layers": 20,
  "ctx_size": 4096
}
```

## Quick Reference Commands

```powershell
# Check GPU status
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
- [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
- [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
- [CMake Downloads](https://cmake.org/download/)
- [SETUP_MISSING_COMPONENTS.md](./SETUP_MISSING_COMPONENTS.md) - Dependency troubleshooting
