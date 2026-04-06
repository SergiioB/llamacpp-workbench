# llama-webui

> Standalone local web UI for llama.cpp with persistent chats, model management, streaming responses, and hardware-aware runtime tuning.

<!-- Badges -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)

<!-- Topics -->
<!-- Topics: python, llm, llama-cpp, gguf, edge-ai, inference, local-llm, llm-inference, rockchip, rk3588, arm64, fastapi -->

---

## What This Is

`llama-webui` is a standalone local web interface for `llama.cpp` — intentionally **without Ollama**. The serving stack is compiled `llama.cpp`, so KV cache quantization, context length, batch sizing, CPU affinity, GPU layer offload, and model-specific flags remain fully controllable.

This project started as a board-local control surface for testing REAP-pruned Mixture-of-Experts models on a `Radxa ROCK 5B+` with `Rockchip RK3588`. It now serves two purposes:

- A practical remote web UI for loading and serving local GGUF models with `llama.cpp`
- A documented benchmark and tuning harness for constrained ARM boards and more capable desktop machines

---

## Key Features

- **llama.cpp only** — no Ollama, no abstractions, full control
- **Browser-driven** model loading and runtime config editing
- **GPU auto-detection** — CUDA, ROCm, Metal, CPU profiles
- **Streaming responses** via Server-Sent Events (SSE)
- **Persistent chat history** in SQLite
- **Model discovery** from configurable GGUF directories
- **RK3588-tested presets** for fast daytime use and stronger overnight use
- **Cross-platform** — ARM SBCs, desktop Linux, NVIDIA GPUs, Apple Silicon

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/SergiioB/llamacpp-workbench.git
cd llamacpp-workbench

# 2. Build llama.cpp (CUDA example)
./scripts/build_llama_cuda.sh

# 3. Download a GGUF model
# e.g. from HuggingFace: https://huggingface.co/Qwen/Qwen3.5-4B-GGUF

# 4. Install and run
uv venv && source .venv/bin/activate
uv pip install -e .
llama-webui
```

Open `http://<host>:8095` in your browser.

---

## Architecture

```
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
│  │  (SQLite)    │  │  (.env)      │  │                  │  │
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

---

## GPU Backend Auto-Detection

| Backend | Detection | Default Settings |
|---------|-----------|------------------|
| **CUDA** | `nvidia-smi` available | `gpu_layers=99`, `parallel=4`, `batch_size=512` |
| **ROCm** | AMD GPU path | Same as CUDA profile |
| **Metal** | Apple Silicon | Managed by llama.cpp |
| **CPU** | Fallback | `gpu_layers=0`, `parallel=1`, `batch_size=128` |

---

## Tested Hardware

| Component | Specification |
|-----------|---------------|
| Board | Radxa ROCK 5B+ |
| SoC | Rockchip RK3588 |
| Architecture | aarch64 |
| Total cores | 8 (4x little + 4x big) |
| RAM | 24 GiB |
| Tested models | GLM-4.7-Flash-REAP-23B-A3B, Qwen3.5-4B |

See [docs/rk3588-benchmarks.md](./docs/rk3588-benchmarks.md) for detailed benchmarks.

---

## Best Configurations

### Best quality-per-size on RK3588

**Model:** `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`

**Settings:**
```
cpu_affinity: 4-7
threads: 4
parallel: 1
context: 202752
batch_size: 128
KV cache: --cache-type-k q8_0 --cache-type-v q4_0
thinking: disabled (--reasoning-budget 0)
```

### Fast daytime fallback

**Model:** `Qwen3.5-4B-Q4_K_M.gguf` — lighter footprint, faster and more reliable for interactive use.

---

## Repository Layout

```
llama-webui/
├── README.md
├── pyproject.toml
├── .env.example
├── .github/              # GitHub Actions CI/CD
├── docs/
│   ├── hardware-portability.md
│   ├── reap-roadmap.md
│   └── rk3588-benchmarks.md
├── models/               # GGUF models (gitignored)
├── data/                 # SQLite, logs (gitignored)
├── scripts/
│   ├── build_llama_cuda.sh
│   ├── build_llama_cuda.ps1
│   ├── setup_windows.ps1
│   ├── reap_benchmark.py
│   └── run_reap_pipeline.sh
├── src/llama_webui/
│   ├── main.py           # FastAPI app entry
│   ├── app_state.py      # SQLite state management
│   ├── llama_manager.py  # llama.cpp process management
│   ├── model_inventory.py # GGUF discovery
│   ├── download_manager.py # Model download handling
│   └── settings.py       # Configuration
└── static/
    ├── app.js            # Frontend logic
    ├── index.html        # Chat UI
    └── styles.css
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_WEBUI_DATA_DIR` | Data directory | `./data` |
| `LLAMA_WEBUI_MODEL_DIRS` | GGUF search paths | `./models`, `~/models` |
| `LLAMA_WEBUI_DEFAULT_DOWNLOAD_DIR` | Model download target | First existing model root |
| `LLAMA_WEBUI_LLAMA_SERVER` | Path to llama-server | Auto-resolved |
| `LLAMA_WEBUI_LLAMA_CLI` | Path to llama-cli | Auto-resolved |

See [.env.example](./.env.example) for concrete examples.

---

## Building llama.cpp

### CUDA Build (NVIDIA GPU)

```bash
./scripts/build_llama_cuda.sh
```

### CPU-only / RK3588 Build

```bash
cd third_party/llama.cpp
mkdir build-rk-opt && cd build-rk-opt
cmake .. -DGGML_CPU_VULKAN=ON -DGGML_CPU_BLAS=ON -DCMAKE_BUILD_TYPE=Release -march=armv8.2a+dotprod
cmake --build . --config Release -j$(nproc)
```

### TurboQuant (TQ3_1S) for 16GB GPUs

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git third_party/llama.cpp-tq3
cd third_party/llama.cpp-tq3
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

---

## What We Learned

- REAP-pruned MoE models deliver surprisingly strong quality-per-size on constrained hardware
- **Disabling reasoning scratchpad** gave dramatically better interactive latency on RK3588
- KV cache quantization (`--cache-type-k q8_0 --cache-type-v q4_0`) matters enough to keep enabled by default
- For this board, the winning configuration was not "enable more thinking" — it was the opposite

---

## Current Status

| Feature | Status |
|---------|--------|
| Web UI | ✅ Production-ready |
| Persistent chats | ✅ Production-ready |
| Model scanning/loading | ✅ Production-ready |
| Streaming responses | ✅ Production-ready |
| Benchmark helpers | ✅ Production-ready |
| RK3588-tuned presets | ✅ Production-ready |
| Corpus ingestion | 🔜 Planned |
| Retrieval/memory pages | 🔜 Planned |
| Personalized REAP build | 🔜 Planned |

---

## Contributing

Contributions welcome. Please:

1. Fork the repo
2. Create a feature branch
3. Run `ruff check . && mypy src/`
4. Submit a PR

---

## License

MIT

---

## Related Links

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [TurboQuant paper (ICLR 2026)](https://arxiv.org/abs/XXXXX)
- [GLM-4.7-Flash-REAP-23B-A3B on HuggingFace](https://huggingface.co/cerebras/GLM-4.7-Flash-REAP-23B-A3B)
- [My portfolio: sergiiob.dev](https://sergiiob.dev)
