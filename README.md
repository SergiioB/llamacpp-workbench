# llama-webui

> Standalone local web UI for `llama.cpp` with persistent chats, model management, streaming responses, and hardware-aware runtime tuning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)

## What This Is

`llama-webui` is a standalone local web interface for `llama.cpp`, intentionally without Ollama. The serving stack is compiled `llama.cpp`, so KV cache quantization, context length, batch sizing, CPU affinity, GPU layer offload, and model-specific flags remain fully controllable.

This repository started as a board-local control surface for testing REAP-pruned Mixture-of-Experts models on a `Radxa ROCK 5B+` with `Rockchip RK3588`. It now serves two purposes:

- A practical remote web UI for loading and serving local GGUF models with `llama.cpp`
- A documented benchmark and tuning harness for constrained ARM boards and more capable desktop machines

## Key Features

- `llama.cpp` only, with no Ollama dependency or hidden abstraction layer
- Browser-driven model loading and runtime config editing
- GPU auto-detection for CUDA, ROCm, Metal, and CPU-first profiles
- Streaming responses via Server-Sent Events
- Persistent chat history in SQLite
- Model discovery from configurable GGUF directories
- Cross-platform support for ARM SBCs, Linux desktops, NVIDIA GPUs, and Windows CUDA hosts
- RK3588-tested presets for fast daytime use and stronger overnight use

## Repository Layout

```text
llama-webui/
├── README.md
├── AGENTS.md
├── CONTRIBUTING.md
├── REAP_RK3588_NOTES.md
├── WINDOWS_SETUP_GUIDE.md
├── SETUP_MISSING_COMPONENTS.md
├── .env.example
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── hardware-portability.md
│   ├── openapi.json
│   ├── reap-roadmap.md
│   └── rk3588-benchmarks.md
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── scripts/
│   ├── build_llama_cuda.sh
│   ├── build_llama_cuda.ps1
│   ├── configure_env.ps1
│   ├── reap_bakeoff.py
│   ├── reap_benchmark.py
│   ├── reap_status.sh
│   ├── run_reap_pipeline.sh
│   └── setup_windows.ps1
├── src/llama_webui/
└── static/
```

Runtime artifacts under `data/`, downloaded models under `models/`, and local virtual environments are intentionally ignored by git.

## Quick Start

### Linux/macOS

1. Clone the repo and enter it.
2. Build `llama.cpp`.
3. Download at least one GGUF into `./models`, `~/models`, or a path listed in `LLAMA_WEBUI_MODEL_DIRS`.
4. Install and run:

```bash
git clone https://github.com/SergiioB/llamacpp-workbench.git
cd llamacpp-workbench

./scripts/build_llama_cuda.sh

uv venv
source .venv/bin/activate
uv pip install -e .
llama-webui
```

### Windows

1. Run the setup script:

```powershell
# Build from source
.\scripts\setup_windows.ps1

# Or use prebuilt binaries
.\scripts\setup_windows.ps1 -UsePrebuilt
```

2. Configure environment variables if needed:

```powershell
.\scripts\configure_env.ps1 -CreateEnvFile
```

3. Start the app:

```powershell
.venv\Scripts\Activate.ps1
llama-webui
```

See [WINDOWS_SETUP_GUIDE.md](./WINDOWS_SETUP_GUIDE.md) and [SETUP_MISSING_COMPONENTS.md](./SETUP_MISSING_COMPONENTS.md) for the Windows flow and dependency troubleshooting.

Open `http://<host>:8095`.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      Browser (Client)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  llama-webui  │  Chat UI  │  Model Settings  │ Logs │   │
│  └───────────────┴───────────┴──────────────────┴──────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────┐
│                     FastAPI Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Chat Router  │  │Model Manager │  │Download Manager  │  │
│  │   (SSE)      │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  App State   │  │   Settings   │  │ Model Inventory  │  │
│  │  (SQLite)    │  │   (.env)     │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ Subprocess / IPC
┌────────────────────────────▼────────────────────────────────┐
│                    llama.cpp (Binary)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │llama-server  │  │  llama-cli   │  │ Custom Flags     │   │
│  │ (HTTP API)   │  │ (Interactive)│  │ gpu_layers, ctx, │   │
│  │              │  │              │  │ threads, etc.    │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

The app is path-portable. Machine-specific paths are handled through environment-driven defaults.

Supported variables:

- `LLAMA_WEBUI_DATA_DIR`
- `LLAMA_WEBUI_MODEL_DIRS`
- `LLAMA_WEBUI_DEFAULT_DOWNLOAD_DIR`
- `LLAMA_WEBUI_LLAMA_SERVER`
- `LLAMA_WEBUI_LLAMA_CLI`

See [.env.example](./.env.example) for concrete examples.

Default behavior without overrides:

- Data directory: `./data`
- Model search roots: `./models` and `~/models`
- Default download target: first existing model root, otherwise `./models`
- `llama-server` and `llama-cli`: resolved from env, then `PATH`, then common local build locations under `third_party/llama.cpp`

## Building llama.cpp

### CUDA Build (NVIDIA GPU)

#### Linux/macOS

```bash
./scripts/build_llama_cuda.sh

# Manual fallback
git clone https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
cd third_party/llama.cpp
mkdir build-cuda && cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_CLI=ON
cmake --build . --config Release -j$(nproc)
```

#### Windows

```powershell
# Recommended
.\scripts\setup_windows.ps1

# Build-only path
.\scripts\build_llama_cuda.ps1

# Prebuilt binaries
.\scripts\setup_windows.ps1 -UsePrebuilt -CudaVersion 12.4
```

Prerequisites for Windows CUDA builds:

- Visual Studio 2022 Build Tools with the C++ desktop workload
- CUDA Toolkit 12.4 or later, with 12.8+ recommended for RTX 50-series
- CMake 3.18+

### CPU-only / RK3588 Build

```bash
cd third_party/llama.cpp
mkdir build-rk-opt && cd build-rk-opt
cmake .. -DGGML_CPU_VULKAN=ON -DGGML_CPU_BLAS=ON -DCMAKE_BUILD_TYPE=Release -march=armv8.2a+dotprod
cmake --build . --config Release -j$(nproc)
```

### TurboQuant (TQ3_1S) for 16GB GPUs

[TurboQuant](https://github.com/turbo-tan/llama.cpp-tq3) enables 3.5-bit quantization to fit larger models on 16GB GPUs with a better size-to-quality tradeoff than plain Q4 variants.

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git third_party/llama.cpp-tq3
cd third_party/llama.cpp-tq3
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

## GPU Backend Auto-Detection

| Backend | Detection | Default Settings |
|---------|-----------|------------------|
| **CUDA** | `nvidia-smi` available | `gpu_layers=99`, `parallel=4`, `batch_size=512` |
| **ROCm** | AMD GPU path | Same as CUDA profile |
| **Metal** | Apple Silicon | Managed by llama.cpp |
| **CPU** | Fallback | `gpu_layers=0`, `parallel=1`, `batch_size=128` |

The detected backend affects default `gpu_layers`, `parallel`, `batch_size`, `ubatch_size`, model presets, and build-path priority for `llama-server`.

## Tested Hardware

### ARM / Embedded

- Board: `Radxa ROCK 5B+`
- SoC: `Rockchip RK3588`
- Architecture: `aarch64`
- Total cores: `8`
- RAM class: about `24 GiB`
- Tested models: `GLM-4.7-Flash-REAP-23B-A3B`, `Qwen3.5-4B`

### Windows Desktop / Laptop

- OS: Windows 11
- CPU: Intel Core Ultra 9 285H
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU plus Intel Arc 140T
- RAM: 32 GB
- CUDA: 12.8+ for Blackwell-class support

See [docs/rk3588-benchmarks.md](./docs/rk3588-benchmarks.md) and [docs/hardware-portability.md](./docs/hardware-portability.md).

## Hardware Scope

Although validation is strongest on the ROCK 5B+, the project is not ARM-only. The UI works anywhere `llama.cpp` works, including:

- ARM SBCs running CPU-only inference
- Desktop Linux CPU-only systems
- NVIDIA CUDA hosts using GPU offload
- AMD ROCm hosts
- Apple Silicon systems
- Windows CUDA laptops and desktops

What changes per machine is not the app architecture, but the runtime tuning: `gpu_layers`, `ctx_size`, `threads`, `parallel`, `batch_size`, `ubatch_size`, and custom `llama.cpp` flags.

## Best Configurations We Found

### Best interactive quality-per-size on RK3588

Model:

- `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`

Validated settings:

- CPU affinity: `4-7`
- Threads: `4`
- Parallel: `1`
- Context: `202752`
- Batch size: `128`
- KV cache quantization: `--cache-type-k q8_0 --cache-type-v q4_0`
- Thinking disabled: `--reasoning-budget 0 --reasoning-format none`

### Fast daytime fallback

- `Qwen3.5-4B-Q4_K_M.gguf`

Reason:

- Much lighter footprint
- Faster and more reliable for interactive use
- Still coherent enough for day-to-day local work

## What We Learned

- REAP-pruned MoE models deliver strong quality-per-size on constrained hardware
- Disabling reasoning scratchpad improved interactive latency materially on RK3588
- KV cache quantization mattered enough to keep enabled by default
- On this board, the best interactive profile came from better runtime tuning, not from enabling more model thinking

## Current Status

| Feature | Status |
|---------|--------|
| Web UI | ✅ Production-ready |
| Persistent chats | ✅ Production-ready |
| Model scanning/loading | ✅ Production-ready |
| Streaming responses | ✅ Production-ready |
| Benchmark helpers | ✅ Production-ready |
| RK3588-tuned presets | ✅ Production-ready |
| Windows setup flow | ✅ Available |
| Corpus ingestion | 🔜 Planned |
| Retrieval/memory pages | 🔜 Planned |
| Personalized REAP build | 🔜 Planned |

## Contributing

Contributions are welcome.

1. Fork the repo
2. Create a feature branch
3. Run `ruff check . && mypy src/`
4. Submit a PR

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidance.

## License

MIT

## Related Links

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [docs/architecture.md](./docs/architecture.md)
- [docs/rk3588-benchmarks.md](./docs/rk3588-benchmarks.md)
- [GLM-4.7-Flash-REAP-23B-A3B on Hugging Face](https://huggingface.co/cerebras/GLM-4.7-Flash-REAP-23B-A3B)
- [Qwen3.5-4B-GGUF on Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B-GGUF)
