from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .settings import GPU_BACKEND, IS_RK3588, model_roots

LEGACY_DEFAULT_MODEL_FILENAMES: set[str] = set()


def _name(model_path: str) -> str:
    return Path(model_path).name.lower()


def _rk3588_profile() -> dict[str, Any]:
    if IS_RK3588:
        return {"cpu_mask": "4-7", "threads": 4}
    detected_cpus = os.cpu_count() or 4
    return {"cpu_mask": "", "threads": max(1, min(4, detected_cpus))}


def _rk3588_custom_args() -> str:
    if IS_RK3588:
        return "--cache-type-k q8_0 --cache-type-v q4_0 --reasoning off --reasoning-budget 0 --reasoning-format none"
    return "--cache-type-k q8_0 --cache-type-v q4_0"


def _is_glm_flash_reap(model_path: str) -> bool:
    name = _name(model_path)
    return "glm-4.7" in name and "flash" in name and "reap" in name and "a3b" in name


def _is_day_qwen(model_path: str) -> bool:
    name = _name(model_path)
    return "qwen" in name and "3.5" in name and "4b" in name and "q4_k_m" in name


def _is_qwen_27b(model_path: str) -> bool:
    return "qwen" in _name(model_path) and "27b" in _name(model_path)


def _find_matching_model(candidates: list[str], predicate: Callable[[str], bool]) -> str | None:
    matches = [candidate for candidate in candidates if predicate(candidate)]
    if not matches:
        return None
    return max(matches, key=model_sort_key)


def list_candidate_models() -> list[str]:
    candidates: list[str] = []
    for root in model_roots():
        if not root.exists():
            continue
        candidates.extend(str(path) for path in root.rglob("*.gguf"))
    return sorted(set(candidates), key=model_sort_key, reverse=True)


def model_sort_key(model_path: str) -> tuple[int, int, int, int, str]:
    name = _name(model_path)
    size_score = 0
    for size_hint, score in (
        ("120b", 120),
        ("72b", 72),
        ("35b", 35),
        ("32b", 32),
        ("28b", 28),
        ("27b", 27),
        ("24b", 24),
        ("23b", 23),
        ("19b", 19),
        ("18b", 18),
        ("16b", 16),
        ("14b", 14),
        ("9b", 9),
        ("8b", 8),
        ("7b", 7),
        ("4b", 4),
        ("3b", 3),
        ("2b", 2),
        ("1.7b", 1),
        ("0.8b", 0),
    ):
        if size_hint in name:
            size_score = score
            break

    a3b_score = 1 if "a3b" in name else 0
    reap_score = 1 if "reap" in name or "prune" in name or "pruned" in name else 0
    quant_score = 1 if "q4_k_m" in name or "q4_k_xl" in name else 0
    return (a3b_score, reap_score, size_score, quant_score, name)


def select_preferred_model(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=model_sort_key)


def normalize_model_path(model_path: str | None, candidates: list[str]) -> str:
    preferred = select_preferred_model(candidates)
    if preferred is None:
        return model_path or ""
    if not model_path:
        return preferred
    if Path(model_path).name.lower() in LEGACY_DEFAULT_MODEL_FILENAMES:
        return preferred
    if not Path(model_path).exists():
        return preferred
    return model_path


def scan_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_path in list_candidate_models():
        path = Path(model_path)
        size_bytes = path.stat().st_size if path.exists() else 0
        name = path.name
        lower = name.lower()
        models.append(
            {
                "path": model_path,
                "name": name,
                "size_bytes": size_bytes,
                "size_gib": round(size_bytes / (1024 ** 3), 2),
                "is_reap": "reap" in lower or "prune" in lower or "pruned" in lower,
                "is_a3b": "a3b" in lower,
                "sort_key": model_sort_key(model_path),
            }
        )
    return sorted(models, key=lambda model: model["sort_key"], reverse=True)


def _apply_cpu_model_profile(
    tuned: dict[str, Any],
    *,
    ctx_size: int,
    batch_size: int = 128,
    ubatch_size: int = 32,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repeat_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> dict[str, Any]:
    profile = _rk3588_profile()
    profile.update(
        {
            "ctx_size": ctx_size,
            "parallel": 1,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "max_tokens": max_tokens,
            "custom_args": _rk3588_custom_args(),
        }
    )
    if temperature is not None:
        profile["temperature"] = temperature
    if top_p is not None:
        profile["top_p"] = top_p
    if top_k is not None:
        profile["top_k"] = top_k
    if repeat_penalty is not None:
        profile["repeat_penalty"] = repeat_penalty
    if presence_penalty is not None:
        profile["presence_penalty"] = presence_penalty
    tuned.update(profile)
    return tuned


def apply_model_profile(config: dict[str, Any], model_path: str) -> dict[str, Any]:
    tuned = {**config, "model_path": model_path}
    lower = _name(model_path)

    if GPU_BACKEND == "cuda":
        return _cuda_profile_for_model(tuned, lower, model_path)

    if _is_glm_flash_reap(model_path):
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=202752,
            batch_size=128,
            ubatch_size=32,
            max_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.0,
            presence_penalty=0.0,
        )

    if _is_qwen_27b(model_path) or "27b" in lower:
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=4096,
            batch_size=128,
            ubatch_size=32,
            max_tokens=1024,
        )

    if "9b" in lower or "8b" in lower or "7b" in lower:
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=8192,
            batch_size=256,
            ubatch_size=64,
            max_tokens=1024,
        )

    if "4b" in lower:
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=16384,
            batch_size=256,
            ubatch_size=64,
            max_tokens=1024,
        )

    if any(size_hint in lower for size_hint in ("3b", "2b", "1.7b")):
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=32768,
            batch_size=256,
            ubatch_size=64,
            max_tokens=1024,
        )

    if "0.8b" in lower:
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=16384,
            batch_size=256,
            ubatch_size=64,
            max_tokens=1024,
        )

    return tuned


def _cuda_profile_for_model(config: dict[str, Any], lower: str, model_path: str) -> dict[str, Any]:
    if _is_qwen_27b(model_path) or "27b" in lower:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 32768,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0",
            }
        )
        return config

    if "72b" in lower or "70b" in lower:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 8192,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0",
            }
        )
        return config

    if "35b" in lower or "34b" in lower or "32b" in lower:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 16384,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0",
            }
        )
        return config

    if "18b" in lower or "17b" in lower or "14b" in lower or "13b" in lower:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 32768,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0",
            }
        )
        return config

    if "9b" in lower or "8b" in lower or "7b" in lower:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 32768,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0",
            }
        )
        return config

    if any(size_hint in lower for size_hint in ("4b", "3b", "2b", "1.7b", "0.8b")):
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 65536,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 4096,
                "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0",
            }
        )
        return config

    config.update(
        {
            "gpu_layers": 99,
            "parallel": 4,
            "batch_size": 512,
            "ubatch_size": 256,
        }
    )
    return config


def build_model_presets(defaults: dict[str, Any], candidates: list[str] | None = None) -> list[dict[str, Any]]:
    available = list(candidates or list_candidate_models())
    presets: list[dict[str, Any]] = []
    day_model = _find_matching_model(available, _is_day_qwen)
    night_model = _find_matching_model(available, _is_glm_flash_reap)

    if GPU_BACKEND == "cuda":
        cuda_large = _find_matching_model(available, lambda p: "27b" in p.lower() or "qwen" in p.lower())
        if cuda_large:
            presets.append(
                {
                    "id": "cuda-qwen-27b",
                    "label": "CUDA Qwen 27B",
                    "description": "High-quality CUDA profile for Qwen 27B on NVIDIA GPU.",
                    "config": apply_model_profile(
                        {
                            **defaults,
                            "temperature": 0.8,
                            "top_p": 0.95,
                            "top_k": 40,
                            "min_p": 0.0,
                            "repeat_penalty": 1.05,
                            "presence_penalty": 0.0,
                        },
                        cuda_large,
                    ),
                }
            )
        return presets

    if day_model:
        presets.append(
            {
                "id": "day-fast",
                "label": "Day Fast 4B",
                "description": "Fast interactive profile for the RK3588 big cores.",
                "config": apply_model_profile(
                    {
                        **defaults,
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "top_k": 40,
                        "min_p": 0.0,
                        "repeat_penalty": 1.05,
                        "presence_penalty": 0.0,
                    },
                    day_model,
                ),
            }
        )

    if night_model:
        presets.append(
            {
                "id": "night-glm-reap",
                "label": "Night GLM REAP",
                "description": "Higher-quality REAP A3B profile for long unattended runs.",
                "config": apply_model_profile(
                    {
                        **defaults,
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 40,
                        "min_p": 0.0,
                        "repeat_penalty": 1.0,
                        "presence_penalty": 0.0,
                    },
                    night_model,
                ),
            }
        )

    return presets
