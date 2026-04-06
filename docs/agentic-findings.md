# Agentic Findings on RK3588

**Date:** 2026-03-30 to 2026-03-31
**Hardware:** Radxa ROCK 5B+ (RK3588) — CPU-only inference

## Test Method

1. Native tool-call test via the OpenAI-compatible `chat.completions` endpoint
2. Direct code-fix test: return the corrected full `calculator.py`, write it to disk, run `pytest`
3. Aider end-to-end test: point Aider at the local `llama.cpp` endpoint and ask it to fix the same failing repo

## Measured Results

### `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`

- Context: `202752`, `threads=4`, `batch=128`, `ubatch=32`
- KV cache: `q8_0` for K, `q4_0` for V
- Native tool call: **success** — `add_numbers(a=2, b=3)` in 69.9s
- Direct code fix: **success** — passed pytest in 69.4s
- Aider end-to-end: **not successful** — too slow for repeated agent loops on CPU-only RK3588

> Excellent chat and raw coding model, native function calling works. Not the best fit for low-latency Aider-style agent loops on this board.

### `qwen3.5-4b-q4_k_m.gguf`

- Context: `65536`, `threads=4`, `batch=256`, `ubatch=64`
- Native tool call: **success** — in 42.8s
- Direct code fix: **success** — in 54.9s
- Aider end-to-end: **success** — in 161.9s, pytest passed

> Best validated agentic option on RK3588 today. Weaker general chat than GLM, but strongest measured fit for existing local coding runtime.

### `qwen3-coder-reap-25b-a3b-q4_k_m.gguf`

- Native tool calls emitted, but arguments were wrong in the benchmark

## Best Current Picks

| Task | Model |
|------|-------|
| General chat / reasoning | `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf` |
| Local coding / agent runtime | `qwen3.5-4b-q4_k_m.gguf` |
| Middle-ground coding quality | `Qwen3.5-9B-Q4_K_M.gguf` |

## Key Insight

On RK3588 CPU-only, the bottleneck is **prompt prefill cost**, not just model size. Practical rules:

- Small dense models can be genuinely usable for local coding loops
- Larger REAP models feel strong but become expensive on large coding-prefill workloads
- Direct constrained edits are much more realistic than generating large correct files from scratch
- Long raw code generation must always be verified

## Scripts

- `scripts/agentic_bakeoff.py` — full agentic bakeoff harness
- `scripts/compare_models_coding.py` — model coding comparison
