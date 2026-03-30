# llama-webui

`llama-webui` is a standalone local web interface for `llama.cpp` with persistent chats, model management, streaming responses, and hardware-aware runtime tuning.

This project intentionally does not use Ollama. The serving stack is compiled `llama.cpp`, so KV cache quantization, context length, batch sizing, CPU affinity, GPU layer offload, and model-specific flags remain fully controllable.

## Why This Exists

This repository started as a board-local control surface for testing REAP-pruned Mixture-of-Experts models on a `Radxa ROCK 5B+` with `Rockchip RK3588`. It now serves two purposes:

- a practical remote web UI for loading and serving local GGUF models with `llama.cpp`
- a documented benchmark and tuning harness for constrained ARM boards and more capable desktop machines

The longer-term goal is to serve a personalized `Qwen3.5-35B-A3B` REAP build derived from exported personal AI session data, but this repository is the serving and benchmarking layer, not the pruning implementation itself.

## Highlights

- `llama.cpp` only: no Ollama dependency
- browser-driven model loading and config editing
- remote bind support on `0.0.0.0`
- model discovery from configurable GGUF directories
- direct GGUF download jobs from the UI
- persistent chat history in SQLite
- streamed chat deltas rendered as markdown
- stop/cancel generation support
- RK3588-tested presets for fast daytime use and stronger overnight use
- portable path configuration for other Linux hosts and desktop GPU systems

## Repository Layout

```text
llama-webui/
├── README.md
├── REAP_RK3588_NOTES.md
├── pyproject.toml
├── .env.example
├── docs/
│   ├── hardware-portability.md
│   ├── reap-roadmap.md
│   └── rk3588-benchmarks.md
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── scripts/
│   ├── reap_bakeoff.py
│   ├── reap_benchmark.py
│   ├── reap_status.sh
│   └── run_reap_pipeline.sh
├── src/llama_webui/
│   ├── app_state.py
│   ├── download_manager.py
│   ├── llama_manager.py
│   ├── main.py
│   ├── model_inventory.py
│   └── settings.py
└── static/
    ├── app.js
    ├── index.html
    ├── styles.css
    └── vendor/markdown-it.min.js
```

Runtime artifacts under `data/`, downloaded models under `models/`, and local virtual environments are intentionally ignored by git.

## Quick Start

1. Build or install `llama.cpp` so `llama-server` and `llama-cli` are available.
2. Download at least one GGUF into `./models`, `~/models`, or any directory listed in `LLAMA_WEBUI_MODEL_DIRS`.
3. Install and run:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
llama-webui
```

Open `http://<host>:8095`.

## Configuration

The app is path-portable. Machine-specific paths have been replaced with environment-driven defaults.

Supported environment variables:

- `LLAMA_WEBUI_DATA_DIR`
- `LLAMA_WEBUI_MODEL_DIRS`
- `LLAMA_WEBUI_DEFAULT_DOWNLOAD_DIR`
- `LLAMA_WEBUI_LLAMA_SERVER`
- `LLAMA_WEBUI_LLAMA_CLI`

See [.env.example](./.env.example) for concrete examples.

Default behavior without overrides:

- data directory: `./data`
- model search roots: `./models` and `~/models`
- default download target: first existing model root, otherwise `./models`
- `llama-server` / `llama-cli`: resolved from env, then `PATH`, then common local build locations under `third_party/llama.cpp`

## Tested Hardware

This repository was validated primarily on:

- board: `Radxa ROCK 5B+`
- SoC: `Rockchip RK3588`
- architecture: `aarch64`
- total cores: `8`
- little cluster: CPUs `0-3`, max `1800 MHz`
- big cluster: CPUs `4-5`, max `2304 MHz`
- faster big cores: CPUs `6-7`, max `2352 MHz`
- RAM class: about `24 GiB`
- swap enabled during testing: about `11 GiB`

The detailed benchmark write-up is in [docs/rk3588-benchmarks.md](./docs/rk3588-benchmarks.md).

## Hardware Scope

Although the strongest validation is on the ROCK 5B+, the repository itself is not ARM-only.

The web UI works anywhere `llama.cpp` works, including:

- ARM SBCs running CPU-only inference
- desktop Linux CPU-only systems
- NVIDIA CUDA hosts using `--n-gpu-layers`
- AMD ROCm hosts using `llama.cpp` GPU offload
- Apple Silicon or other systems, as long as the local `llama.cpp` build and binaries are available

What changes per machine is not the app architecture, but the runtime tuning:

- `gpu_layers`
- `ctx_size`
- `threads`
- `parallel`
- `batch_size`
- `ubatch_size`
- custom `llama.cpp` flags

See [docs/hardware-portability.md](./docs/hardware-portability.md) for starting profiles on non-RK3588 hardware.

## Models We Actually Tested

The most important real board-local candidates were:

- `cerebras/GLM-4.7-Flash-REAP-23B-A3B`
- `Qwen/Qwen3.5-4B-GGUF`

Links:

- <https://huggingface.co/cerebras/GLM-4.7-Flash-REAP-23B-A3B>
- <https://huggingface.co/zai-org/GLM-4.7-Flash>
- <https://huggingface.co/Qwen/Qwen3.5-4B-GGUF>

## Best Configurations We Found

### Best interactive quality-per-size on the ROCK 5B+

Model:

- `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`

Settings that were validated locally:

- CPU affinity: `4-7`
- threads: `4`
- parallel: `1`
- context: `202752`
- batch size: `128`
- ubatch size: `32`
- temperature: `1.0`
- top-p: `0.95`
- top-k: `40`
- max tokens: `1024` interactive default
- KV cache quantization: `--cache-type-k q8_0 --cache-type-v q4_0`
- thinking: disabled via `--reasoning-budget 0 --reasoning-format none`

### Best fast daytime fallback on the ROCK 5B+

Model:

- `Qwen3.5-4B-Q4_K_M.gguf`

Reason:

- much lighter footprint
- clearly faster and more reliable than larger REAP candidates on the board
- still coherent enough for day-to-day interactive use

The detailed evidence is in [docs/rk3588-benchmarks.md](./docs/rk3588-benchmarks.md).

## What We Learned About REAP On This Board

- REAP-pruned MoE models can deliver surprisingly strong quality-per-size on RK3588.
- The winning local configuration was not "enable more thinking"; it was the opposite.
- For this board, disabling reasoning scratchpad gave dramatically better interactive latency.
- KV cache quantization mattered enough to keep enabled by default.
- Full trained context on the GLM REAP model was usable from a serving perspective, but prefill and total latency still have to be judged against workload shape.

## Personal Data / Corpus Workflow

This repository does not ship personal exports or a pruning dataset.

The intended external workflow is:

1. export assistant and web-chat history with a separate extraction tool
2. normalize those exports into a pruning or retrieval corpus
3. run pruning or further analysis off-device
4. bring the resulting GGUF back here for benchmarking and serving

The public-safe roadmap and expected data layout are documented in [docs/reap-roadmap.md](./docs/reap-roadmap.md).

## Current Status

What is production-usable now:

- web UI
- persistent chats
- model scanning and loading
- streaming responses with markdown rendering
- benchmark helpers
- RK3588-tuned presets

What is still missing:

- first-class corpus ingestion inside the UI
- retrieval / memory pages
- integrated REAP job orchestration
- a personalized `Qwen3.5-35B-A3B` artifact produced from a real user corpus

## Notes

- This repository has intentionally been stripped of host-specific absolute paths.
- Runtime databases, logs, downloaded models, and virtual environments are ignored by git.
- License selection is intentionally left to the repository owner rather than assumed here.
