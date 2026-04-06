#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


PROMPT = """You are benchmarking a local model on resource-constrained hardware.

Write a structured technical note for an engineer who wants to run a large Mixture-of-Experts model on a small ARM board.

Cover:
1. What determines prefill speed.
2. What determines generation speed.
3. Why KV cache quantization matters.
4. How model quantization affects quality and RAM.
5. How to avoid swap thrashing.
6. How to choose between a faster weaker model and a slower stronger model.
7. When a pruned REAP MoE is preferable to a dense model.
8. Practical tuning steps for llama.cpp on CPU-only hardware.

Use detailed prose and numbered sections."""

PROMPT_RE = re.compile(r"prompt eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")
GEN_RE = re.compile(r"\beval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")
TOTAL_RE = re.compile(r"total time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")


def _benchmark_dir() -> Path:
    from llama_webui.settings import benchmark_dir

    return benchmark_dir()


def _resolve_llama_cli_binary() -> str:
    from llama_webui.settings import resolve_llama_cli_binary

    return resolve_llama_cli_binary()


def read_rss_kib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def kill_process_tree(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark REAP model candidates on RK3588.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpu-mask", default="4-7")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ubatch-size", type=int, default=32)
    parser.add_argument("--predict", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--llama-cli", default=_resolve_llama_cli_binary())
    parser.add_argument("--output-dir", default=str(_benchmark_dir()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "taskset",
        "-c",
        args.cpu_mask,
        args.llama_cli,
        "-m", args.model,
        "-t", str(args.threads),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ctk", "q8_0",
        "-ctv", "q4_0",
        "--temp", "1.0",
        "--top-p", "0.95",
        "--top-k", "40",
        "-n", str(args.predict),
        "-p", PROMPT,
        "--simple-io",
        "--no-display-prompt",
        "--perf",
    ]
    cmd.extend(args.extra_arg)

    started = time.time()
    max_rss_kib = 0
    timed_out = False
    output = ""
    stop_sampling = threading.Event()
    proc: subprocess.Popen[str] | None = None

    with tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        errors="replace",
        delete=False,
        prefix=f"{args.label}-",
        suffix=".log",
    ) as tmp_output:
        tmp_output_path = Path(tmp_output.name)
        proc = subprocess.Popen(
            cmd,
            stdout=tmp_output,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )

        def sample_rss() -> None:
            nonlocal max_rss_kib
            assert proc is not None
            while not stop_sampling.is_set():
                max_rss_kib = max(max_rss_kib, read_rss_kib(proc.pid))
                if proc.poll() is not None:
                    break
                time.sleep(0.25)

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()

        try:
            proc.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(proc.pid)
        finally:
            stop_sampling.set()
            sampler.join(timeout=1)
            if proc.poll() is None:
                kill_process_tree(proc.pid)
            tmp_output.flush()

    output = tmp_output_path.read_text(encoding="utf-8", errors="replace")
    assert proc is not None

    prompt_match = PROMPT_RE.search(output)
    gen_match = GEN_RE.search(output)
    total_match = TOTAL_RE.search(output)

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "model": args.model,
        "threads": args.threads,
        "cpu_mask": args.cpu_mask,
        "ctx_size": args.ctx_size,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "predict": args.predict,
        "timeout_seconds": args.timeout_seconds,
        "extra_args": args.extra_arg,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "max_rss_mib": round(max_rss_kib / 1024, 1),
        "prompt_eval_seconds": round(float(prompt_match.group(1)) / 1000.0, 3) if prompt_match else None,
        "prompt_eval_tokens": int(prompt_match.group(2)) if prompt_match else None,
        "generation_seconds": round(float(gen_match.group(1)) / 1000.0, 3) if gen_match else None,
        "generation_tokens": int(gen_match.group(2)) if gen_match else None,
        "total_seconds": round(float(total_match.group(1)) / 1000.0, 3) if total_match else round(time.time() - started, 3),
        "total_tokens": int(total_match.group(2)) if total_match else None,
        "success": proc.returncode == 0 and not timed_out and prompt_match is not None and gen_match is not None,
        "output_chars": len(output),
        "output_tail": output[-6000:],
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{timestamp}-{args.label}-1000tok.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
