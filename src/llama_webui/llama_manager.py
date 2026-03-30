from __future__ import annotations

import http.client
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import re
from typing import Any


class GenerationCancelledError(RuntimeError):
    pass


THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class LlamaServerManager:
    def _sanitize_visible_content(self, content: str) -> str:
        cleaned = content.strip()
        if "</think>" in cleaned.lower():
            lower = cleaned.lower()
            marker = lower.rfind("</think>")
            cleaned = cleaned[marker + len("</think>"):].strip()
        cleaned = THINK_TAG_PATTERN.sub("", cleaned).strip()
        if cleaned.lower().startswith("<think>"):
            cleaned = cleaned[len("<think>"):].strip()
        return cleaned

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.pid_path = log_path.with_name("llama-server.pid")
        self.process: subprocess.Popen[bytes] | None = None
        self.started_with: dict[str, Any] | None = None
        self._active_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._active_connection: http.client.HTTPConnection | None = None

    def _base_url(self, config: dict[str, Any]) -> str:
        return f"http://{config['llama_host']}:{config['llama_port']}"

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {error.code}: {body}") from error

    def _managed_pid(self) -> int | None:
        if self.process and self.process.poll() is None:
            return self.process.pid
        try:
            pid = int(self.pid_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                self.pid_path.unlink()
            except FileNotFoundError:
                pass
            return None
        return pid

    def health(self, config: dict[str, Any]) -> dict[str, Any]:
        managed_pid = self._managed_pid()
        try:
            models = self._request("GET", f"{self._base_url(config)}/v1/models", timeout=3.0)
            return {
                "healthy": True,
                "managed": managed_pid is not None,
                "models": models.get("data", []),
                "pid": managed_pid,
            }
        except Exception as error:
            return {
                "healthy": False,
                "managed": managed_pid is not None,
                "models": [],
                "pid": managed_pid,
                "error": str(error),
            }

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        self.stop()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        args: list[str] = []
        cpu_mask = str(config.get("cpu_mask") or "").strip()
        if cpu_mask:
            args.extend(["taskset", "-c", cpu_mask])
        args.extend([
            str(config["llama_binary"]),
            "--host", str(config["llama_host"]),
            "--port", str(config["llama_port"]),
            "--model", str(config["model_path"]),
            "--ctx-size", str(config["ctx_size"]),
            "--threads", str(config["threads"]),
            "--parallel", str(config["parallel"]),
            "--n-gpu-layers", str(config["gpu_layers"]),
            "--batch-size", str(config["batch_size"]),
            "--ubatch-size", str(config["ubatch_size"]),
        ])
        if str(config.get("custom_args") or "").strip():
            args.extend(shlex.split(str(config["custom_args"])))

        log_handle = self.log_path.open("ab")
        self.process = subprocess.Popen(args, stdout=log_handle, stderr=log_handle)
        self.pid_path.write_text(f"{self.process.pid}\n")
        self.started_with = dict(config)

        deadline = time.time() + 180
        last_error = "llama-server did not become healthy"
        while time.time() < deadline:
            status = self.health(config)
            if status["healthy"]:
                return status
            if self.process.poll() is not None:
                last_error = f"llama-server exited with code {self.process.returncode}"
                break
            last_error = status.get("error", last_error)
            time.sleep(1)
        raise RuntimeError(last_error)

    def stop(self) -> None:
        self.stop_generation()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        else:
            managed_pid = self._managed_pid()
            if managed_pid is not None:
                try:
                    os.kill(managed_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.process = None
        self.started_with = None
        try:
            self.pid_path.unlink()
        except FileNotFoundError:
            pass

    def stop_generation(self) -> bool:
        with self._active_lock:
            cancel = self._active_cancel
            connection = self._active_connection
            if cancel is None:
                return False
            cancel.set()
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            return True

    def chat(self, config: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
        result: dict[str, Any] | None = None
        for event in self.chat_stream(config, messages):
            if event["type"] == "done":
                result = event
        if result is None:
            raise RuntimeError("llama.cpp stream ended without a completion result")
        return result

    def chat_stream(self, config: dict[str, Any], messages: list[dict[str, str]]):
        payload = {
            "model": "local",
            "messages": messages,
            "stream": True,
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "min_p": config["min_p"],
            "repeat_penalty": config["repeat_penalty"],
            "presence_penalty": config["presence_penalty"],
            "max_tokens": config["max_tokens"],
        }
        started = time.perf_counter()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        cancel = threading.Event()
        connection = http.client.HTTPConnection(
            str(config["llama_host"]),
            int(config["llama_port"]),
            timeout=300,
        )
        with self._active_lock:
            if self._active_cancel is not None:
                raise RuntimeError("generation already in progress")
            self._active_cancel = cancel
            self._active_connection = connection

        response: http.client.HTTPResponse | None = None
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            response = connection.getresponse()
            if response.status >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"llama.cpp HTTP {response.status}: {body}")

            while True:
                if cancel.is_set():
                    raise GenerationCancelledError("generation cancelled")
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break

                event = json.loads(data)
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    content_parts.append(piece)
                    yield {"type": "delta", "delta": piece, "content": "".join(content_parts)}
                reasoning_piece = delta.get("reasoning_content")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
        except Exception as error:
            if not cancel.is_set():
                raise RuntimeError(str(error)) from error
        finally:
            try:
                if response is not None:
                    response.close()
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
            with self._active_lock:
                self._active_cancel = None
                self._active_connection = None

        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        content = self._sanitize_visible_content(content)
        if not content and reasoning:
            # Keep hidden scratchpad out of the visible response path when the
            # model decides to emit reasoning_content despite a zero budget.
            content = ""
        yield {
            "type": "done",
            "content": content,
            "cancelled": cancel.is_set(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
