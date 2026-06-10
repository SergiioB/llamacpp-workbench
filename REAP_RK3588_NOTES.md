# REAP RK3588 Notes

This file remains the tracked RK3588 note in the repo.

The longer internal write-ups may live elsewhere, but the main board-local takeaways worth keeping in the repository are:

- `llama-webui` is intentionally usable on `Radxa ROCK 5B+` / `RK3588`, not only on desktop GPUs
- CPU inference should stay pinned to the big cores with `taskset -c 4-7`
- explicit no-thinking startup defaults are the right interactive default on this board:
  - `--reasoning off`
  - `--reasoning-budget 0`
  - `--reasoning-format none`
- `Qwen3.5-2B` is the practical fast CPU tier for interactive use
- `Qwen3.5-4B` is slower, but still the better quality tier when latency can be traded for structure
- even with server-side reasoning disabled, some models can still leak empty `<think>` wrappers, so visible-response sanitation is still worth keeping in the UI layer

For the broader project description and positioning, see:

- [README.md](./README.md)
