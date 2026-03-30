from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"


def _split_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def data_dir() -> Path:
    configured = os.environ.get("LLAMA_WEBUI_DATA_DIR")
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "data"


def model_roots() -> tuple[Path, ...]:
    configured = _split_paths(os.environ.get("LLAMA_WEBUI_MODEL_DIRS"))
    if configured:
        return tuple(configured)
    return (
        PROJECT_ROOT / "models",
        Path.home() / "models",
    )


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
    return _resolve_binary(
        "LLAMA_WEBUI_LLAMA_SERVER",
        "llama-server",
        [
            PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server",
            PROJECT_ROOT / "third_party" / "llama.cpp" / "build-rk-opt" / "bin" / "llama-server",
        ],
    )


def resolve_llama_cli_binary() -> str:
    return _resolve_binary(
        "LLAMA_WEBUI_LLAMA_CLI",
        "llama-cli",
        [
            PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli",
            PROJECT_ROOT / "third_party" / "llama.cpp" / "build-rk-opt" / "bin" / "llama-cli",
        ],
    )
