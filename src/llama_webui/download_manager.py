from __future__ import annotations

import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _tail_lines(path: Path, max_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    lines: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip())
    return list(lines)


class ModelDownloadManager:
    def __init__(self, log_dir: Path, default_destination_dir: Path) -> None:
        self.log_dir = log_dir
        self.default_destination_dir = default_destination_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def _resolve_destination(self, url: str, destination_path: str | None) -> Path:
        if destination_path and destination_path.strip():
            return Path(destination_path).expanduser()

        parsed = urlparse(url)
        basename = Path(parsed.path).name or "model.gguf"
        if "?" in basename:
            basename = basename.split("?", 1)[0]
        return self.default_destination_dir / basename

    def start_download(self, url: str, destination_path: str | None = None) -> dict[str, Any]:
        destination = self._resolve_destination(url, destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex[:12]
        log_path = self.log_dir / f"download-{job_id}.log"
        log_handle = log_path.open("ab")
        command = ["wget", "-c", "-O", str(destination), url]
        process = subprocess.Popen(command, stdout=log_handle, stderr=log_handle)

        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "url": url,
                "destination_path": str(destination),
                "log_path": str(log_path),
                "command": command,
                "process": process,
            }
        return self.get_job(job_id)

    def _snapshot(self, job: dict[str, Any]) -> dict[str, Any]:
        process: subprocess.Popen[bytes] = job["process"]
        returncode = process.poll()
        if returncode is None:
            status = "running"
        elif returncode == 0:
            status = "completed"
        else:
            status = "failed"

        destination = Path(job["destination_path"])
        log_path = Path(job["log_path"])
        return {
            "job_id": job["job_id"],
            "url": job["url"],
            "destination_path": job["destination_path"],
            "status": status,
            "pid": process.pid if returncode is None else None,
            "returncode": returncode,
            "downloaded_bytes": destination.stat().st_size if destination.exists() else 0,
            "log_path": job["log_path"],
            "log_tail": _tail_lines(log_path),
            "command": job["command"],
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [self._snapshot(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job["job_id"], reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Download job {job_id} not found")
            return self._snapshot(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Download job {job_id} not found")
            process: subprocess.Popen[bytes] = job["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            return self._snapshot(job)
