# Hardware Portability

This repository was validated primarily on a `Radxa ROCK 5B+`, but the project itself is not tied to that board. It is a generic `llama.cpp` web UI with a board-specific set of proven defaults.

The important distinction is:

- validated profile: exact settings that were measured on the ROCK 5B+
- portable profile: starting points for other machines that still need local tuning

## What Is Board-Specific

The following findings are specific to the tested ROCK 5B+ / RK3588 environment:

- pinning `llama.cpp` to CPUs `4-7` was beneficial
- `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf` was the best quality-per-size result
- `Qwen3.5-4B-Q4_K_M.gguf` was the safest fast fallback
- `--cache-type-k q8_0 --cache-type-v q4_0` was a good default
- disabling reasoning was much better than enabling it for interactive use

Do not assume those exact winning values will also be optimal on a workstation GPU.

## What Transfers Cleanly To Other Hardware

The application architecture is portable:

- `llama-server` process control
- GGUF model scanning
- persistent chats
- remote web UI
- browser-side streaming
- direct `llama.cpp` flag control
- browser-based inference via WebGPU (no llama.cpp build needed)

That means the same repo can run on:

- ARM boards
- x86 CPU-only systems
- NVIDIA CUDA hosts
- AMD ROCm hosts
- Apple Silicon, if `llama.cpp` is built and available in the local environment

## Auto-Detection

The application auto-detects your GPU backend on startup:

- **CUDA**: NVIDIA GPUs with `nvidia-smi` available
- **ROCm**: AMD GPUs with ROCm support
- **Metal**: Apple Silicon
- **CPU**: Fallback when no GPU acceleration is detected

The detected backend affects:
- Default `gpu_layers` (99 for CUDA, 0 for CPU)
- Default `parallel`, `batch_size`, `ubatch_size` values
- Model presets shown in the UI
- Build path priority for `llama-server` binary

## Building llama.cpp

### CUDA Build (NVIDIA)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build-cuda && cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
# Binary ends up in: bin/llama-server
```

The application automatically prefers `third_party/llama.cpp/build-cuda/bin/llama-server` on CUDA systems.

### CPU-only Build (RK3588 / ARM)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build-rk-opt && cd build-rk-opt
cmake .. -DGGML_CPU_VULKAN=ON -DGGML_CPU_BLAS=ON -DCMAKE_BUILD_TYPE=Release -march=armv8.2a+dotprod
cmake --build . --config Release -j$(nproc)
# Binary ends up in: bin/llama-server
```

The RK3588-optimized build uses Vulkan for GPU offload (Mali) and OpenBLAS for CPU acceleration.

## Practical Starting Profiles

These are starting points, not validated final answers.

### ARM SBC / CPU-only

Use when:

- RAM is limited
- there is no practical `llama.cpp` GPU offload path

Starting point:

- `gpu_layers=0`
- set `threads` to your performance-core count, not total logical threads by default
- if the SoC is big.LITTLE, pin to the big cluster when possible
- start with `batch_size=128`
- start with `ubatch_size=32`
- keep KV quantization enabled
- use smaller dense models or compact REAP variants first

### Desktop CPU-only

Use when:

- you have many CPU cores and enough RAM
- GPU offload is unavailable or undesirable

Starting point:

- `gpu_layers=0`
- `threads` near physical core count
- `parallel=1`
- `batch_size=256` or `512`
- `ubatch_size=64` or `128`
- larger contexts are often practical, but validate prefill cost separately

### NVIDIA / CUDA

Use when:

- the model or part of the model fits comfortably in VRAM

Starting point:

- offload with `gpu_layers=99` to keep the hot path on GPU
- raise `parallel=4`, `batch_size=512`, `ubatch_size=256`
- keep an eye on VRAM headroom before increasing context aggressively
- retest context and max tokens once offload is stable

On these machines, the best model may be very different from the ROCK 5B+ winner because the tradeoff changes from CPU throughput to VRAM fit and PCIe / host-device transfer behavior.

### TurboQuant (TQ3_1S) on CUDA

The [TurboQuant TQ3_1S format](https://github.com/turbo-tan/llama.cpp-tq3) enables 3.5-bit quantization that fits 27B models on 16GB GPUs:

- TQ3_1S is ~10% smaller than Q4_0 with near-identical quality
- Works on NVIDIA GPUs with CUDA support
- See the [original tweet](https://x.com/coffeecup2020/status/2038725930626003140) for benchmarks

To use TQ3_1S models:

1. Build llama.cpp with TurboQuant support from the [fork](https://github.com/turbo-tan/llama.cpp-tq3)
2. Download TQ3_1S quantized models from [HuggingFace](https://huggingface.co/YTan2000/Qwen3.5-27B-TQ3_1S)
3. Place in your models directory
4. Load via the web UI - the CUDA profile will handle it

### AMD / ROCm

Use when:

- you have working `llama.cpp` ROCm support

Starting point:

- same general approach as CUDA: offload as much as fits cleanly
- benchmark prompt speed and generation speed separately
- validate the effect of larger `batch_size` and `ubatch_size`

### Apple Silicon

Use when:

- `llama.cpp` Metal acceleration is available

Starting point:

- do not copy the RK3588 CPU pinning strategy
- treat unified memory and Metal offload as a different tuning problem
- start with higher context and batch values than the SBC profile, then measure

## Tuning Order On Any New Machine

When porting this repo to another host, tune in this order:

1. confirm `llama-server` and `llama-cli` paths (see build scripts in `scripts/`)
2. confirm model scanning and loading
3. verify one short chat round-trip
4. tune `gpu_layers`
5. tune `threads`
6. tune `batch_size`
7. tune `ubatch_size`
8. increase context only after load stability is proven
9. test reasoning on and off separately

## Recommended Documentation Discipline

When benchmarking a new machine, keep two classes of settings separate:

- measured settings: actually benchmarked on that hardware
- suggested settings: reasonable starting points that still need validation

This repository already follows that split for the ROCK 5B+ documentation, and the same pattern should be used for desktop GPU results later.
