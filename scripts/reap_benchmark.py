#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llama_webui.settings import benchmark_dir, resolve_llama_cli_binary


PROMPT = "Say hello in one short sentence."
SPEED_RE = re.compile(r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s")


def read_meminfo() -> dict[str, int]:
    data: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if not parts:
            continue
        data[key] = int(parts[0])
    return data


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
    parser = argparse.ArgumentParser(description="Benchmark a local GGUF model on RK3588 with llama.cpp.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--contexts", default="2048,4096,8192,16384")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ubatch-size", type=int, default=32)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--llama-cli", default=resolve_llama_cli_binary())
    parser.add_argument("--output-dir", default=str(benchmark_dir()))
    return parser.parse_args()


def run_one(ctx: int, args: argparse.Namespace) -> dict[str, Any]:
    before = read_meminfo()
    cmd = [
        args.llama_cli,
        "-m", args.model,
        "-t", str(args.threads),
        "-c", str(ctx),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ctk", "q8_0",
        "-ctv", "q4_0",
        "-n", str(args.predict),
        "-p", PROMPT,
        "-st",
        "--simple-io",
        "--no-display-prompt",
        "--perf",
    ]

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )

    max_rss_kib = 0
    timed_out = False
    output = ""
    try:
        while proc.poll() is None:
            max_rss_kib = max(max_rss_kib, read_rss_kib(proc.pid))
            if time.time() - started > args.timeout_seconds:
                timed_out = True
                kill_process_tree(proc.pid)
                break
            time.sleep(0.5)
        if proc.stdout is not None:
            output = proc.stdout.read()
    finally:
        if proc.poll() is None:
            kill_process_tree(proc.pid)

    ended = time.time()
    after = read_meminfo()
    match = SPEED_RE.search(output)
    prompt_tps = float(match.group(1)) if match else None
    generation_tps = float(match.group(2)) if match else None

    return {
        "ctx_size": ctx,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "duration_seconds": round(ended - started, 3),
        "max_rss_mib": round(max_rss_kib / 1024, 1),
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "mem_before": {
            "mem_available_mib": round(before.get("MemAvailable", 0) / 1024, 1),
            "swap_free_mib": round(before.get("SwapFree", 0) / 1024, 1),
        },
        "mem_after": {
            "mem_available_mib": round(after.get("MemAvailable", 0) / 1024, 1),
            "swap_free_mib": round(after.get("SwapFree", 0) / 1024, 1),
        },
        "success": bool(match) and proc.returncode == 0 and not timed_out,
        "output_tail": output[-4000:],
    }


def main() -> int:
    args = parse_args()
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "model": str(model_path),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "predict": args.predict,
        "prompt": PROMPT,
        "contexts": [],
    }

    for ctx in contexts:
        run = run_one(ctx, args)
        result["contexts"].append(run)
        if not run["success"]:
            break

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{timestamp}-{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
