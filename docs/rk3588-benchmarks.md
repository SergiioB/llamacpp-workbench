# RK3588 Benchmarks And Tuning

This document captures the board-local findings that were actually validated while serving and benchmarking `llama.cpp` models on a `Radxa ROCK 5B+` with `Rockchip RK3588`.

## Test Platform

`lscpu` on the target board reports:

- board model: `Radxa ROCK 5B+`
- architecture: `aarch64`
- SoC: `Rockchip RK3588`
- total cores: `8`
- little cluster: CPUs `0-3`, max `1800 MHz`
- big cluster: CPUs `4-5`, max `2304 MHz`
- faster big cores: CPUs `6-7`, max `2352 MHz`
- cache: `384 KiB` L1d, `384 KiB` L1i, `2.5 MiB` L2, `3 MiB` L3

Memory profile during this work:

- RAM class: about `24 GiB`
- swap configured: about `11 GiB`

For the best interactive results on this board, `llama.cpp` was pinned to the big cores with `taskset -c 4-7`.

## Serving Stack

Everything here used compiled `llama.cpp`, not Ollama.

Key runtime choices:

- server: `llama-server`
- benchmark CLI: `llama-cli`
- KV cache quantization: `q8_0` for K, `q4_0` for V
- web UI default bind: `0.0.0.0:8095`
- local `llama.cpp` API bind: `127.0.0.1:8085`

## Winning Model On This Board

### `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`

Model family:

- base serving model: `cerebras/GLM-4.7-Flash-REAP-23B-A3B`
- board-tested GGUF quant: `Q3_K_M`

Why it won:

- much stronger than the small dense baselines
- surprisingly responsive for its effective strength on this board
- better quality-per-size than the larger REAP candidates we attempted locally

Best validated interactive config:

- CPU mask: `4-7`
- threads: `4`
- parallel: `1`
- ctx size: `202752`
- batch size: `128`
- ubatch size: `32`
- temperature: `1.0`
- top-p: `0.95`
- top-k: `40`
- min-p: `0.0`
- repeat penalty: `1.0`
- presence penalty: `0.0`
- interactive max tokens: `1024`
- custom args: `--cache-type-k q8_0 --cache-type-v q4_0 --reasoning-budget 0 --reasoning-format none`

## Reasoning Mode: What Actually Worked

The official GLM guidance allows thinking mode, but the best local result on RK3588 was not "low thinking". It was full disable.

Observed locally:

- thinking off, full `202752` context
  - smoke prompt wall time: about `1.02 s`
  - slang prompt wall time: about `2.61 s`
- thinking on with the same model and context
  - smoke prompt total time: about `18.74 s`
  - slang prompt: exceeded `120 s` and was aborted

Important implementation detail:

- the `llama.cpp` build used during this work supported `--reasoning-budget 0` or `-1`
- there was no usable "small reasoning budget" middle ground to tune

Conclusion:

- for this board and this serving mode, disabling visible reasoning and internal reasoning budget was the correct choice for interactive use

## Quantization And Context Choices

### KV cache

The winning setup used:

- `--cache-type-k q8_0`
- `--cache-type-v q4_0`

This kept memory growth in check without noticeably harming the interactive behavior of the tested models.

### Full-context serving

The GLM REAP model reported a trained context of `202752`, and that full setting was accepted and served successfully through `llama-server`.

That does not mean every workload will be pleasant at full window size. It means:

- the model can be loaded and served at that context on this board
- prefill cost still has to be judged per workload
- large-context overnight jobs and smaller interactive jobs should be treated differently

## Benchmarks With Real Outputs

### GLM REAP rerun benchmark

Source artifact:

- `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`

Saved benchmark result:

- `ctx_size=2048`
- `threads=8`
- `batch_size=128`
- `ubatch_size=32`
- prompt speed: `2.6 t/s`
- generation speed: `0.8 t/s`
- total duration: `143.279 s`
- peak RSS: `5292.9 MiB`
- swap pressure: severe by the end of the run

This run mattered as a cautionary baseline, but it was not the final serving config because the reasoning path and thread shape were later improved for the web UI.

### Qwen3.5 4B dense baseline

Saved benchmark result:

- model: `Qwen3.5-4B-Q4_K_M.gguf`
- `ctx_size=512`
- `threads=8`
- `batch_size=128`
- `ubatch_size=32`
- prompt speed: `4.5 t/s`
- generation speed: `3.7 t/s`
- total duration: `31.542 s`
- peak RSS: `5372.1 MiB`

Why it remains important:

- it is the best low-risk daytime fallback
- it stays in the repo presets because it is reliable on constrained hardware

## Models We Tried And Rejected

These were not retained as recommended defaults for this board:

- larger 18B REAP variants: too much memory pressure or poor interactive behavior
- 14B MXFP4 REAP variant: incompatible with the tested `llama.cpp` build here
- dense 27B fallback as the main answer: useful as a comparison point, but not the best balance versus the REAP GLM result

The key lesson is not that bigger models never work on RK3588. It is that board-local viability depends on the combined effect of:

- quantization
- active parameters
- CPU affinity
- reasoning mode
- batch and ubatch sizing
- swap avoidance

## Recommended Model Strategy

For an RK3588 deployment similar to this one:

- best interactive quality-per-size: `GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf`
- best fast fallback: `Qwen3.5-4B-Q4_K_M.gguf`

Recommended operating split:

- day profile: smaller dense fallback for quick, low-risk use
- night profile: GLM REAP with large context for stronger unattended runs
