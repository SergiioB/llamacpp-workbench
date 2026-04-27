from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"


def _split_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def _has_nvidia_gpu() -> bool:
    if os.path.exists("/proc/driver/nvidia"):
        return True
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5)
        return result.returncode == 0 and b"GPU" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _detect_gpu_backend() -> Literal["cuda", "rocm", "metal", "cpu"]:
    if _has_nvidia_gpu():
        return "cuda"
    if os.path.exists("/sys/kernel/mm/amd-tee"):
        return "rocm"
    if os.uname().machine.startswith("arm64") and os.path.exists("/System/Library/Extensions/AGL.framework"):
        return "metal"
    return "cpu"


def _is_rk3588() -> bool:
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compatible = f.read().decode("ascii", errors="replace").lower()
            return "rk3588" in compatible
    except (FileNotFoundError, OSError, PermissionError):
        return False


GPU_BACKEND: Literal["cuda", "rocm", "metal", "cpu"] = _detect_gpu_backend()
IS_RK3588: bool = _is_rk3588()


def data_dir() -> Path:
    configured = os.environ.get("LLAMA_WEBUI_DATA_DIR")
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "data"


def model_roots() -> tuple[Path, ...]:
    configured = _split_paths(os.environ.get("LLAMA_WEBUI_MODEL_DIRS"))
    # Auto-discover HuggingFace cache and common model directories. Explicit
    # LLAMA_WEBUI_MODEL_DIRS entries are additive so launch scripts do not hide
    # models in ~/models or ~/llama-rpc/models.
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    fallbacks: list[Path] = list(configured)
    if hf_cache.exists():
        for model_dir in hf_cache.glob("models--*"):
            snapshot_dir = model_dir / "snapshots"
            if snapshot_dir.exists():
                for snapshot in snapshot_dir.iterdir():
                    if snapshot.is_dir():
                        fallbacks.append(snapshot)
    fallbacks.extend([
        PROJECT_ROOT / "models",
        Path.home() / "models",
        Path.home() / "llama-rpc" / "models",
    ])
    return tuple(fallbacks)


def default_download_dir() -> Path:
    configured = os.environ.get("LLAMA_WEBUI_DEFAULT_DOWNLOAD_DIR")
    if configured:
        return Path(configured).expanduser()
    for root in model_roots():
        if root.exists():
            return root
    return model_roots()[0]


def benchmark_dir() -> Path:
    return data_dir() / "benchmarks"


def _resolve_binary(env_var: str, command_name: str, candidates: list[Path]) -> str:
    configured = os.environ.get(env_var)
    if configured:
        return str(Path(configured).expanduser())

    from_path = shutil.which(command_name)
    if from_path:
        return from_path

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return command_name


def resolve_llama_server_binary() -> str:
    base = PROJECT_ROOT / "third_party" / "llama.cpp"
    candidates = [
        base / "prebuilt" / "llama-server.exe",  # Windows prebuilt
        base / "prebuilt" / "llama-server",
        base / "build-cuda" / "bin" / "llama-server",
        base / "build" / "bin" / "llama-server",
        base / "build-rk-opt" / "bin" / "llama-server",
    ]
    if GPU_BACKEND == "cuda":
        cuda_first = [
            base / "build-cuda" / "bin" / "llama-server",
            base / "prebuilt" / "llama-server.exe",
            base / "prebuilt" / "llama-server",
            base / "build" / "bin" / "llama-server",
        ]
        candidates = cuda_first + [c for c in candidates if c not in cuda_first]
    return _resolve_binary("LLAMA_WEBUI_LLAMA_SERVER", "llama-server", candidates)


def resolve_llama_cli_binary() -> str:
    base = PROJECT_ROOT / "third_party" / "llama.cpp"
    candidates = [
        base / "prebuilt" / "llama-cli.exe",  # Windows prebuilt
        base / "prebuilt" / "llama-cli",
        base / "build-cuda" / "bin" / "llama-cli",
        base / "build" / "bin" / "llama-cli",
        base / "build-rk-opt" / "bin" / "llama-cli",
    ]
    if GPU_BACKEND == "cuda":
        cuda_first = [
            base / "build-cuda" / "bin" / "llama-cli",
            base / "prebuilt" / "llama-cli.exe",
            base / "prebuilt" / "llama-cli",
            base / "build" / "bin" / "llama-cli",
        ]
        candidates = cuda_first + [c for c in candidates if c not in cuda_first]
    return _resolve_binary("LLAMA_WEBUI_LLAMA_CLI", "llama-cli", candidates)
