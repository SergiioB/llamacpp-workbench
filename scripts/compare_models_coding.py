#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from agentic_bakeoff import (
    MODELS,
    RESULTS_DIR,
    direct_code_fix_test,
    reset_sandbox,
    start_server,
    stop_process,
    wait_for_health,
)
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_SERVER = Path(
    os.environ.get("LLAMA_WEBUI_LLAMA_SERVER")
    or "/home/radxa/projects/intelliauto-discord-bot/third_party/llama.cpp/build-rk-opt/bin/llama-server"
)
TARGET_PREFILL_CHARS = 6000
TARGET_PREFILL_MAX_TOKENS = 180
TARGET_LONG_CODE_MAX_TOKENS = 700
CLIENT_TIMEOUT_SECONDS = 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local coding models on the same workloads.")
    parser.add_argument("--models", nargs="*", default=["glm_reap_23b", "qwen_9b", "qwen_4b"])
    parser.add_argument("--base-port", type=int, default=18210)
    parser.add_argument("--llama-server", default=str(DEFAULT_LLAMA_SERVER))
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_large_context_prompt() -> str:
    files = [
        ROOT / "scripts" / "agentic_bakeoff.py",
        ROOT / "docs" / "2026-03-30-rk3588-findings.md",
        ROOT / "README.md",
    ]
    sections = []
    for path in files:
        sections.append(f"FILE: {path.name}\n{read_text(path)}")
    bundle = "\n\n".join(sections)
    if len(bundle) < TARGET_PREFILL_CHARS:
        bundle = (bundle + "\n\n") * (TARGET_PREFILL_CHARS // max(len(bundle), 1) + 1)
    return (
        "You are reading a local benchmark workspace.\n"
        "Use only the context below.\n\n"
        f"{bundle[:TARGET_PREFILL_CHARS]}\n\n"
        "Task:\n"
        "Answer in exactly 3 bullet points.\n"
        "1. What board was tested?\n"
        "2. Which model is currently the best validated local agent-runtime model?\n"
        "3. Why are GLM REAP and Qwen 4B split into different recommended roles?\n"
    )


def build_long_code_prompt() -> str:
    return (
        "Write only Python code, no markdown fences.\n"
        "Create a complete single-file module named `workspace_index.py` that implements:\n"
        "- a dataclass `ChunkHit` with path, score, snippet, start_line, end_line\n"
        "- a class `WorkspaceIndex` with methods:\n"
        "  - `add_document(path: str, text: str) -> None`\n"
        "  - `remove_document(path: str) -> None`\n"
        "  - `search(query: str, limit: int = 5) -> list[ChunkHit]`\n"
        "  - `to_json() -> str`\n"
        "  - `from_json(data: str) -> WorkspaceIndex`\n"
        "- chunk text by paragraphs and also by fixed fallback windows\n"
        "- lexical scoring using token overlap and path boosting\n"
        "- a small CLI in `main()` that can add files from a folder and search them\n"
        "- type hints throughout\n"
        "- docstrings on public classes/functions\n"
        "- no external dependencies\n"
        "Make the implementation non-trivial and complete.\n"
    )


def build_ansible_generation_prompt() -> str:
    return (
        "Write only YAML, no markdown fences.\n"
        "Create a complete Ansible playbook named `site.yml` for Debian/Ubuntu web servers.\n"
        "Requirements:\n"
        "- play targets group `webservers`\n"
        "- uses `become: true`\n"
        "- defines vars `web_root`, `motd_message`, and `nginx_server_name`\n"
        "- updates apt cache safely\n"
        "- installs `nginx` and `ufw`\n"
        "- ensures `/srv/www/app` exists with owner/group `www-data`\n"
        "- manages `/etc/hosts` with `blockinfile`\n"
        "- writes `/etc/motd` with `copy`\n"
        "- deploys an nginx site config to `/etc/nginx/sites-available/app.conf`\n"
        "- symlinks it into `sites-enabled`\n"
        "- reloads nginx through a handler\n"
        "- enables and starts nginx\n"
        "- opens `OpenSSH`, `Nginx Full` in ufw\n"
        "- includes tags on major tasks\n"
        "- idempotent and reasonably professional\n"
    )


def build_ansible_fix_prompt() -> str:
    broken = """---
- name: Configure web servers
  hosts: webservers
  gather_facts: no
  vars:
    web_root: /srv/www/app
  tasks:
    - name: Install packages
      apt:
      name:
        - nginx
        - ufw
      state: present

    - name: Add host entry
      lineinfile:
        path: /etc/hosts
        line: \"192.168.1.100 app.internal\"

    - name: Deploy nginx config
      copy:
        dest: /etc/nginx/sites-available/app.conf
        content: |
          server {
            listen 80;
            server_name _;
            root {{ web_root }};
          }
      notify: reload nginx

    - name: Enable nginx site
      file:
        src: /etc/nginx/sites-available/app.conf
        dest: /etc/nginx/sites-enabled/app.conf
        state: link

    - name: Ensure nginx running
      service:
        name: nginx
        state: started
        enabled: true
  handlers:
  - name: reload nginx
    service:
      name: nginx
      state: reloaded
"""
    return (
        "You are fixing a broken Ansible playbook.\n"
        "Return only the complete corrected YAML for `site.yml`.\n"
        "Make it valid, idempotent, and professional.\n"
        "Ensure it uses become, apt cache update, a handler, and safe hosts management.\n\n"
        f"{broken}"
    )


TIMING_RE = re.compile(
    r"prompt eval time =\s+(?P<prompt_ms>[0-9.]+) ms / \s*(?P<prompt_tokens>\d+) tokens.*?\n"
    r"\s*eval time =\s+(?P<eval_ms>[0-9.]+) ms / \s*(?P<eval_tokens>\d+) tokens.*?\n"
    r"\s*total time =\s+(?P<total_ms>[0-9.]+) ms / \s*(?P<total_tokens>\d+) tokens",
    re.DOTALL,
)

MEMORY_RE = {
    "cpu_mapped_mib": re.compile(r"CPU_Mapped model buffer size =\s+([0-9.]+) MiB"),
    "cpu_repack_mib": re.compile(r"CPU_REPACK model buffer size =\s+([0-9.]+) MiB"),
    "cpu_kleidiai_mib": re.compile(r"CPU_KLEIDIAI model buffer size =\s+([0-9.]+) MiB"),
    "kv_cache_mib": re.compile(r"CPU KV buffer size =\s+([0-9.]+) MiB"),
    "compute_buffer_mib": re.compile(r"CPU compute buffer size =\s+([0-9.]+) MiB"),
}


class ProcSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.max_rss_mib = 0.0
        self.max_cpu_percent = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join(timeout=2)
        return {
            "peak_rss_mib": round(self.max_rss_mib, 1),
            "peak_cpu_percent": round(self.max_cpu_percent, 1),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                status = Path(f"/proc/{self.pid}/status").read_text(encoding="utf-8", errors="replace")
                rss_match = re.search(r"VmRSS:\s+(\d+)\s+kB", status)
                if rss_match:
                    self.max_rss_mib = max(self.max_rss_mib, int(rss_match.group(1)) / 1024)
                proc = subprocess.run(
                    ["ps", "-p", str(self.pid), "-o", "%cpu="],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                cpu_text = proc.stdout.strip()
                if cpu_text:
                    self.max_cpu_percent = max(self.max_cpu_percent, float(cpu_text))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)


def parse_latest_timing(log_path: Path, previous_count: int) -> tuple[dict[str, Any] | None, int]:
    text = read_text(log_path) if log_path.exists() else ""
    matches = list(TIMING_RE.finditer(text))
    if len(matches) <= previous_count:
        return None, len(matches)
    match = matches[-1]
    return (
        {
            "prompt_eval_seconds": round(float(match.group("prompt_ms")) / 1000, 3),
            "prompt_tokens": int(match.group("prompt_tokens")),
            "generation_seconds": round(float(match.group("eval_ms")) / 1000, 3),
            "generation_tokens": int(match.group("eval_tokens")),
            "total_seconds_from_log": round(float(match.group("total_ms")) / 1000, 3),
            "total_tokens_from_log": int(match.group("total_tokens")),
        },
        len(matches),
    )


def parse_memory_summary(log_path: Path) -> dict[str, float]:
    text = read_text(log_path) if log_path.exists() else ""
    result: dict[str, float] = {}
    for key, pattern in MEMORY_RE.items():
        match = pattern.search(text)
        if match:
            result[key] = round(float(match.group(1)), 2)
    return result


def run_chat(
    client: OpenAI,
    log_path: Path,
    timing_count: int,
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model="local",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = round(time.perf_counter() - started, 3)
    content = response.choices[0].message.content or ""
    timing, timing_count = parse_latest_timing(log_path, timing_count)
    return (
        {
            "latency_seconds": elapsed,
            "content": content,
            "content_chars": len(content),
            "timing": timing,
        },
        timing_count,
    )


def extract_python_code(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def compile_python(code: str) -> dict[str, Any]:
    if not code:
        return {"compile_ok": False, "error": "empty output"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "workspace_index.py"
        path.write_text(code + "\n", encoding="utf-8")
        proc = subprocess.run(
            ["python3", "-m", "py_compile", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "compile_ok": proc.returncode == 0,
            "error": (proc.stderr or proc.stdout)[-4000:],
        }


def extract_yaml(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    if cleaned.lower().startswith("site.yml"):
        cleaned = cleaned.split("\n", 1)[1].lstrip()
    return cleaned.strip()


def validate_ansible_yaml(text: str) -> dict[str, Any]:
    cleaned = extract_yaml(text)
    if not cleaned:
        return {"yaml_ok": False, "error": "empty output"}
    try:
        parsed = yaml.safe_load(cleaned)
    except Exception as exc:  # noqa: BLE001
        return {
            "yaml_ok": False,
            "error": str(exc),
            "cleaned_preview": cleaned[:1200],
        }

    plays = parsed if isinstance(parsed, list) else [parsed]
    serialized = json.dumps(parsed)
    text_lower = cleaned.lower()
    return {
        "yaml_ok": True,
        "play_count": len(plays),
        "has_become": "become: true" in text_lower,
        "has_handlers": "handlers:" in text_lower,
        "has_blockinfile": "blockinfile:" in text_lower,
        "has_ufw": "ufw:" in text_lower,
        "has_notify": "notify:" in text_lower,
        "has_copy": "copy:" in text_lower,
        "has_service": "service:" in text_lower,
        "mentions_nginx": "nginx" in serialized.lower(),
    }


def judge_prefill_answer(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "mentions_board": "rock 5b" in lowered,
        "mentions_qwen4b": "qwen3.5-4b-q4_k_m" in lowered or "qwen 4b" in lowered,
        "mentions_glm_role_split": ("glm" in lowered and "chat" in lowered) or ("slower" in lowered and "aider" in lowered),
    }


def score_ansible_validation(validation: dict[str, Any]) -> int:
    if not validation.get("yaml_ok"):
        return 0
    keys = [
        "has_become",
        "has_handlers",
        "has_blockinfile",
        "has_ufw",
        "has_notify",
        "has_copy",
        "has_service",
        "mentions_nginx",
    ]
    return sum(1 for key in keys if validation.get(key))


def compare_model(model_key: str, port: int, llama_server: Path) -> dict[str, Any]:
    spec = MODELS[model_key]
    log_path = RESULTS_DIR / f"coding-compare-{spec.label}.log"
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

    timing_count = 0
    sampler: ProcSampler | None = None
    try:
        server = start_server(llama_server, spec, port, log_path)
        wait_for_health(f"http://127.0.0.1:{port}/v1", 240)
        sampler = ProcSampler(server.pid)
        sampler.start()
        client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local",
            timeout=CLIENT_TIMEOUT_SECONDS,
            max_retries=0,
        )

        prefill_result, timing_count = run_chat(
            client,
            log_path,
            timing_count,
            system="You are a careful codebase analyst.",
            user=build_large_context_prompt(),
            max_tokens=TARGET_PREFILL_MAX_TOKENS,
            temperature=0.0,
        )
        prefill_result["judge"] = judge_prefill_answer(prefill_result["content"])
        result["large_prefill_analysis"] = prefill_result

        long_code_result, timing_count = run_chat(
            client,
            log_path,
            timing_count,
            system="You are a precise senior Python engineer.",
            user=build_long_code_prompt(),
            max_tokens=TARGET_LONG_CODE_MAX_TOKENS,
            temperature=0.0,
        )
        code = extract_python_code(long_code_result["content"])
        long_code_result["compile"] = compile_python(code)
        long_code_result["line_count"] = len(code.splitlines()) if code else 0
        result["long_code_generation"] = long_code_result

        ansible_generation_result, timing_count = run_chat(
            client,
            log_path,
            timing_count,
            system="You are a senior infrastructure engineer writing Ansible.",
            user=build_ansible_generation_prompt(),
            max_tokens=650,
            temperature=0.0,
        )
        ansible_generation_result["validation"] = validate_ansible_yaml(ansible_generation_result["content"])
        ansible_generation_result["validation_score"] = score_ansible_validation(
            ansible_generation_result["validation"]
        )
        result["ansible_generation"] = ansible_generation_result

        ansible_fix_result, timing_count = run_chat(
            client,
            log_path,
            timing_count,
            system="You are a precise Ansible reviewer and editor.",
            user=build_ansible_fix_prompt(),
            max_tokens=500,
            temperature=0.0,
        )
        ansible_fix_result["validation"] = validate_ansible_yaml(ansible_fix_result["content"])
        ansible_fix_result["validation_score"] = score_ansible_validation(ansible_fix_result["validation"])
        result["ansible_fix"] = ansible_fix_result

        sandbox = reset_sandbox()
        result["direct_code_fix_sandbox_dir"] = str(sandbox)
        result["direct_code_fix"] = direct_code_fix_test(f"http://127.0.0.1:{port}/v1", sandbox)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        if log_path.exists():
            result["log_tail"] = read_text(log_path)[-12000:]
    finally:
        if sampler:
            result["runtime_pressure"] = sampler.stop()
        stop_process(server)
        result["memory_summary"] = parse_memory_summary(log_path)
    return result


def main() -> int:
    args = parse_args()
    llama_server = Path(args.llama_server)
    if not llama_server.exists():
        raise SystemExit(f"llama-server not found at {llama_server}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, model_key in enumerate(args.models):
        results.append(compare_model(model_key, args.base_port + index, llama_server))

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = RESULTS_DIR / f"{timestamp}-coding-compare.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
