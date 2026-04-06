#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SANDBOX_TEMPLATE = ROOT / "sandbox_repo"
DEFAULT_MODEL_ROOT = Path(os.environ.get("LLAMA_WEBUI_MODEL_ROOT", str(Path.home() / "models")))
LLAMA_SERVER = Path(
    os.environ.get("LLAMA_WEBUI_LLAMA_SERVER")
    or shutil.which("llama-server")
    or ""
)


@dataclass
class ModelSpec:
    label: str
    path: Path
    ctx_size: int
    threads: int
    batch_size: int
    ubatch_size: int
    custom_args: list[str]


MODELS: dict[str, ModelSpec] = {
    "glm_reap_23b": ModelSpec(
        label="glm_reap_23b",
        path=DEFAULT_MODEL_ROOT / "reap_gguf" / "GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf",
        ctx_size=202752,
        threads=4,
        batch_size=128,
        ubatch_size=32,
        custom_args=["--cache-type-k", "q8_0", "--cache-type-v", "q4_0", "--reasoning-budget", "0", "--reasoning-format", "none"],
    ),
    "qwen_4b": ModelSpec(
        label="qwen_4b",
        path=DEFAULT_MODEL_ROOT / "qwen3.5-4b-q4_k_m.gguf",
        ctx_size=65536,
        threads=4,
        batch_size=256,
        ubatch_size=64,
        custom_args=["--cache-type-k", "q8_0", "--cache-type-v", "q4_0", "--reasoning-budget", "0", "--reasoning-format", "none"],
    ),
    "qwen_9b": ModelSpec(
        label="qwen_9b",
        path=DEFAULT_MODEL_ROOT / "qwen3.5-9b-q4_k_m.gguf",
        ctx_size=65536,
        threads=4,
        batch_size=192,
        ubatch_size=48,
        custom_args=["--cache-type-k", "q8_0", "--cache-type-v", "q4_0", "--reasoning-budget", "0", "--reasoning-format", "none"],
    ),
    "qwen_coder_reap_25b": ModelSpec(
        label="qwen_coder_reap_25b",
        path=DEFAULT_MODEL_ROOT / "agentic_bakeoff" / "qwen3-coder-reap-25b-a3b-q4_k_m.gguf",
        ctx_size=32768,
        threads=4,
        batch_size=128,
        ubatch_size=32,
        custom_args=["--cache-type-k", "q8_0", "--cache-type-v", "q4_0", "--reasoning-budget", "0"],
    ),
    "kimi_linear_reap_35b": ModelSpec(
        label="kimi_linear_reap_35b",
        path=DEFAULT_MODEL_ROOT / "agentic_bakeoff" / "Kimi-Linear-REAP-35B-A3B-Instruct-IQ3_M.gguf",
        ctx_size=16384,
        threads=4,
        batch_size=128,
        ubatch_size=32,
        custom_args=["--cache-type-k", "q8_0", "--cache-type-v", "q4_0"],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic bakeoff for local llama.cpp models.")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--base-port", type=int, default=8091)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--llama-server", default=str(LLAMA_SERVER) if LLAMA_SERVER else "")
    return parser.parse_args()


def wait_for_health(base_url: str, timeout_seconds: int) -> None:
    client = OpenAI(base_url=base_url, api_key="local")
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            client.models.list()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"model did not become healthy: {last_error}")


def stop_process(proc: subprocess.Popen[Any] | None) -> None:
    if not proc:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def safe_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def start_server(llama_server: Path, spec: ModelSpec, port: int, log_path: Path) -> subprocess.Popen[Any]:
    log_handle = log_path.open("wb")
    cmd = [
        "taskset",
        "-c",
        "4-7",
        str(llama_server),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(spec.path),
        "--ctx-size",
        str(spec.ctx_size),
        "--threads",
        str(spec.threads),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "0",
        "--batch-size",
        str(spec.batch_size),
        "--ubatch-size",
        str(spec.ubatch_size),
        *spec.custom_args,
    ]
    return subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)


def tool_call_test(base_url: str) -> dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key="local")
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": "You are a strict tool user."},
                {"role": "user", "content": "Use the add_numbers tool to compute 2 + 3. Do not answer directly."},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two integers together.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=128,
            temperature=0.0,
        )
        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        return {
            "success": bool(tool_calls),
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "content": choice.message.content,
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def direct_code_fix_test(base_url: str, repo_dir: Path) -> dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key="local")
    started = time.perf_counter()
    original = (repo_dir / "calculator.py").read_text(encoding="utf-8")
    prompt = (
        "You are fixing a tiny Python repository.\n"
        "Read the failing test and return the complete corrected contents of calculator.py only.\n"
        "Do not include markdown fences, commentary, or any other files.\n\n"
        "helpers.py:\n"
        f"{(repo_dir / 'helpers.py').read_text(encoding='utf-8')}\n\n"
        "test_calculator.py:\n"
        f"{(repo_dir / 'test_calculator.py').read_text(encoding='utf-8')}\n\n"
        "calculator.py:\n"
        f"{original}\n"
    )
    try:
        response = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": "You are a precise Python code editor."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
        wrote_file = cleaned.startswith("from ") and "summarize_topics" in cleaned
        if wrote_file:
            (repo_dir / "calculator.py").write_text(cleaned + "\n", encoding="utf-8")
        after = pytest_status(repo_dir)
        diff = subprocess.run(["git", "diff", "--", "."], cwd=repo_dir, text=True, capture_output=True, check=True)
        return {
            "success": wrote_file and after.returncode == 0,
            "wrote_file": wrote_file,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "response_text": content[-6000:],
            "pytest_after": {
                "returncode": after.returncode,
                "stdout": after.stdout[-4000:],
                "stderr": after.stderr[-4000:],
            },
            "git_diff": diff.stdout[-12000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def reset_sandbox() -> Path:
    sandbox_parent = ROOT / "results" / "tmp"
    sandbox_parent.mkdir(parents=True, exist_ok=True)
    sandbox_dir = Path(tempfile.mkdtemp(prefix="agentic-repo-", dir=sandbox_parent))
    for item in SANDBOX_TEMPLATE.iterdir():
        target = sandbox_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    subprocess.run(["git", "init", "-b", "main"], cwd=sandbox_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Agentic Bakeoff"], cwd=sandbox_dir, check=True)
    subprocess.run(["git", "config", "user.email", "agentic@example.invalid"], cwd=sandbox_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=sandbox_dir, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=sandbox_dir, check=True, capture_output=True)
    return sandbox_dir


def pytest_status(repo_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repo_dir,
        text=True,
        capture_output=True,
    )


def aider_test(base_url: str, repo_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    before = pytest_status(repo_dir)
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = base_url
    env["OPENAI_API_KEY"] = "local"
    started = time.perf_counter()
    command = [
        str(ROOT / ".venv" / "bin" / "aider"),
        "--model",
        "openai/local",
        "--edit-format",
        "whole",
        "--yes-always",
        "--no-auto-commits",
        "--no-show-model-warnings",
        "--no-check-update",
        "--no-show-release-notes",
        "--no-gitignore",
        "--map-tokens",
        "0",
        "--message",
        "Read the tests and fix the implementation so pytest passes. Make the minimum correct change.",
        "calculator.py",
        "helpers.py",
        "test_calculator.py",
    ]
    aider_stdout = ""
    aider_stderr = ""
    aider_returncode = None
    try:
        proc = subprocess.run(
            command,
            cwd=repo_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        timed_out = False
        aider_stdout = proc.stdout or ""
        aider_stderr = proc.stderr or ""
        aider_returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        aider_stdout = safe_text(exc.stdout)
        aider_stderr = safe_text(exc.stderr)

    after = pytest_status(repo_dir)
    diff = subprocess.run(["git", "diff", "--", "."], cwd=repo_dir, text=True, capture_output=True, check=True)
    return {
        "success": before.returncode != 0 and after.returncode == 0 and not timed_out,
        "timed_out": timed_out,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "pytest_before": {
            "returncode": before.returncode,
            "stdout": before.stdout[-4000:],
            "stderr": before.stderr[-4000:],
        },
        "pytest_after": {
            "returncode": after.returncode,
            "stdout": after.stdout[-4000:],
            "stderr": after.stderr[-4000:],
        },
        "aider_returncode": aider_returncode,
        "aider_stdout": aider_stdout[-12000:],
        "aider_stderr": aider_stderr[-12000:],
        "git_diff": diff.stdout[-12000:],
    }


def benchmark_model(llama_server: Path, spec: ModelSpec, port: int, timeout_seconds: int) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{spec.label}-llama-server.log"
    server = None
    result: dict[str, Any] = {
        "label": spec.label,
        "model_path": str(spec.path),
        "port": port,
        "exists": spec.path.exists(),
    }
    if not spec.path.exists():
        result["error"] = "model file missing"
        return result

    try:
        server = start_server(llama_server, spec, port, log_path)
        wait_for_health(f"http://127.0.0.1:{port}/v1", 240)
        result["server_started"] = True
        result["tool_call_test"] = tool_call_test(f"http://127.0.0.1:{port}/v1")
        direct_repo = reset_sandbox()
        result["direct_code_fix_sandbox_dir"] = str(direct_repo)
        result["direct_code_fix_test"] = direct_code_fix_test(f"http://127.0.0.1:{port}/v1", direct_repo)
        aider_repo = reset_sandbox()
        result["aider_sandbox_dir"] = str(aider_repo)
        result["aider_test"] = aider_test(f"http://127.0.0.1:{port}/v1", aider_repo, timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        if log_path.exists():
            result["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
    finally:
        stop_process(server)
    return result


def main() -> int:
    args = parse_args()
    llama_server = Path(args.llama_server)
    if not args.llama_server:
        raise SystemExit("llama-server not found; set LLAMA_WEBUI_LLAMA_SERVER or pass --llama-server")
    if not llama_server.exists():
        raise SystemExit(f"llama-server not found at {llama_server}")
    run_results = []
    for index, model_key in enumerate(args.models):
        spec = MODELS[model_key]
        run_results.append(benchmark_model(llama_server, spec, args.base_port + index, args.timeout_seconds))

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = RESULTS_DIR / f"{timestamp}-agentic-bakeoff.json"
    out_path.write_text(json.dumps(run_results, indent=2))
    print(json.dumps(run_results, indent=2))
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
