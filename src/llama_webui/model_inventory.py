from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .settings import GPU_BACKEND, IS_RK3588, model_roots

LEGACY_DEFAULT_MODEL_FILENAMES: set[str] = set()

_EXCLUDED_MODEL_PATTERNS = re.compile(r"mmproj", re.IGNORECASE)

_SIZE_HINTS: tuple[tuple[str, int], ...] = (
    ("120b", 120),
    ("72b", 72),
    ("70b", 70),
    ("35b", 35),
    ("34b", 34),
    ("32b", 32),
    ("28b", 28),
    ("27b", 27),
    ("24b", 24),
    ("23b", 23),
    ("19b", 19),
    ("18b", 18),
    ("16b", 16),
    ("14b", 14),
    ("13b", 13),
    ("9b", 9),
    ("8b", 8),
    ("7b", 7),
    ("4b", 4),
    ("3b", 3),
    ("2b", 2),
    ("1.7b", 1),
    ("0.8b", 0),
)


def _name(model_path: str) -> str:
    return Path(model_path).name.lower()


def _is_excluded_model(name: str) -> bool:
    return bool(_EXCLUDED_MODEL_PATTERNS.search(name))


def detect_model_size_billions(model_path: str) -> float | None:
    name = _name(model_path)
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?![a-z])", name)
    if match:
        return float(match.group(1))
    for size_hint, score in _SIZE_HINTS:
        if size_hint in name:
            return float(score)
    return None


def is_moe_model(model_path: str) -> bool:
    name = _name(model_path)
    if "a3b" in name:
        return True
    moe_match = re.search(r"(\d+)b\s*[-_x]\s*(\d+)b", name)
    return bool(moe_match)


def is_reap_model(model_path: str) -> bool:
    name = _name(model_path)
    return "reap" in name or "prune" in name or "pruned" in name


def _is_qwen36(model_path: str) -> bool:
    name = _name(model_path)
    return "qwen3.6" in name or ("qwen" in name and "3.6" in name)


def _is_qwen36_moe_35b(model_path: str) -> bool:
    name = _name(model_path)
    return (
        _is_qwen36(model_path) and "35b" in name and "a3b" in name and not is_reap_model(model_path)
    )


def _is_qwen36_dense_27b(model_path: str) -> bool:
    name = _name(model_path)
    return _is_qwen36(model_path) and "27b" in name and "dflash" not in name


def _is_qwen36_reap_28b(model_path: str) -> bool:
    name = _name(model_path)
    return _is_qwen36(model_path) and "28b" in name and is_reap_model(model_path)


def _is_qwen36_hauhau_uncensored(model_path: str) -> bool:
    name = _name(model_path)
    return _is_qwen36(model_path) and "hauhau" in name and "uncensored" in name


def _is_qwen35_size(model_path: str, size_b: float) -> bool:
    name = _name(model_path)
    return "qwen3.5" in name and detect_model_size_billions(model_path) == size_b


def _validation_metadata(model_path: str) -> dict[str, str]:
    name = _name(model_path)
    if _is_qwen36_moe_35b(model_path) and "ud-iq3_s" in name:
        return {
            "validation_status": "recommended",
            "validation_label": "RPC best",
            "validation_note": "RPC-validated winner: 128K Q8 KV, 39.6 tok/s, quality tests passed.",
        }
    if _is_qwen36_dense_27b(model_path):
        return {
            "validation_status": "fallback",
            "validation_label": "RPC fallback",
            "validation_note": "RPC-validated dense fallback: 64K Q8 KV, 18.5 tok/s, quality tests passed.",
        }
    if _is_qwen36_reap_28b(model_path):
        return {
            "validation_status": "avoid",
            "validation_label": "RPC avoid",
            "validation_note": "RPC-tested fast, but failed factual quality checks after expert pruning.",
        }
    if _is_qwen36_hauhau_uncensored(model_path):
        if "q3_k_p" in name:
            return {
                "validation_status": "rpc_oom",
                "validation_label": "RPC OOM",
                "validation_note": "RPC-tested/reviewed: 18 GiB quant exceeds the practical 20 GiB combined VRAM budget.",
            }
        else:
            return {
                "validation_status": "rpc_constrained",
                "validation_label": "RPC constrained",
                "validation_note": "RPC-tested/reviewed: only fits at constrained 2/8 split with 8K context and is too slow for this setup.",
            }
    if _is_qwen35_size(model_path, 9):
        return {
            "validation_status": "validated",
            "validation_label": "Validated 9B",
            "validation_note": "Validated smaller Qwen quality tier; practical on 12GB-class CUDA.",
        }
    if _is_qwen35_size(model_path, 4):
        return {
            "validation_status": "validated",
            "validation_label": "Validated 4B",
            "validation_note": "Validated working fast tier; fits comfortably and was the known WebUI baseline.",
        }
    if _is_qwen35_size(model_path, 2):
        return {
            "validation_status": "validated",
            "validation_label": "Validated 2B",
            "validation_note": "Validated RK3588 fast CPU tier.",
        }
    if _is_qwen35_size(model_path, 27):
        return {
            "validation_status": "too_large",
            "validation_label": "Too large",
            "validation_note": "Seen in WebUI setup notes as too large for a single 12GB GPU.",
        }
    if _is_glm_flash_reap(model_path):
        return {
            "validation_status": "validated",
            "validation_label": "RK3588 validated",
            "validation_note": "Validated long-running REAP/RK3588 profile.",
        }
    return {
        "validation_status": "unvalidated",
        "validation_label": "Unvalidated",
        "validation_note": "Scanned GGUF; no local benchmark verdict recorded.",
    }


def estimate_vram_gib(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    size_bytes = path.stat().st_size if path.exists() else 0
    file_gib = size_bytes / (1024**3)
    model_size = detect_model_size_billions(model_path)
    moe = is_moe_model(model_path)
    reap = is_reap_model(model_path)

    kv_overhead_per_1k_ctx_gib = 0.0
    if model_size is not None:
        kv_overhead_per_1k_ctx_gib = 0.004 if moe else model_size * 0.00012

    min_vram_gib = file_gib * 1.05
    return {
        "file_gib": round(file_gib, 2),
        "min_vram_gib": round(min_vram_gib, 2),
        "kv_overhead_per_1k_ctx_gib": round(kv_overhead_per_1k_ctx_gib, 4),
        "model_size_b": model_size,
        "is_moe": moe,
        "is_reap": reap,
        "fits_4gb": min_vram_gib <= 4.0,
        "fits_8gb": min_vram_gib <= 8.0,
        "fits_12gb": min_vram_gib <= 12.0,
        "fits_16gb": min_vram_gib <= 16.0,
        "fits_24gb": min_vram_gib <= 24.0,
    }


def suggest_gpu_layers(model_path: str, available_vram_gib: float | None = None) -> int:
    """Suggest GPU layers based on available VRAM.

    The estimate intentionally leaves headroom for display use, CUDA/runtime
    overhead, KV cache, and transient compute buffers.
    """
    if available_vram_gib is not None:
        file_gib = Path(model_path).stat().st_size / (1024**3) if Path(model_path).exists() else 0
        # Leave 1.5GB for CUDA overhead, display, KV cache
        usable = available_vram_gib - 1.5
        ratio = min(1.0, usable / file_gib) if file_gib > 0 else 1.0
        model_size = detect_model_size_billions(model_path)
        if model_size is not None and model_size > 0:
            total_layers = _estimate_total_layers(model_size, is_moe_model(model_path))
            return max(1, int(total_layers * ratio))
        return 99 if ratio >= 0.95 else 50

    # Conservative defaults for consumer GPUs when exact free VRAM is unknown.
    model_size = detect_model_size_billions(model_path)
    if model_size is None:
        return 99
    if model_size >= 70:
        return 30  # Very partial offload
    if model_size >= 27:
        return 35
    if model_size >= 18:
        return 50
    if model_size >= 13:
        return 70
    return 99  # 9B and below fit fully


def _estimate_total_layers(model_size: float, moe: bool) -> int:
    if moe:
        if model_size >= 20:
            return 58
        if model_size >= 8:
            return 36
        return 28
    if model_size >= 60:
        return 80
    if model_size >= 30:
        return 64
    if model_size >= 13:
        return 48
    if model_size >= 6:
        return 36
    return 28


def _rk3588_profile() -> dict[str, Any]:
    if IS_RK3588:
        return {"cpu_mask": "4-7", "threads": 4}
    detected_cpus = os.cpu_count() or 4
    return {"cpu_mask": "", "threads": max(1, min(4, detected_cpus))}


def _platform_custom_args(model_path: str) -> str:
    """Return platform-appropriate custom KV cache args."""
    if IS_RK3588:
        return f"{_kv_cache_args(model_path)} --reasoning off --reasoning-budget 0 --reasoning-format none"
    return _kv_cache_args(model_path)


def _kv_cache_args(_model_path: str) -> str:
    """Choose KV cache quantization based on model size.

    Quantized V cache requires Flash Attention in llama.cpp. Keep both K and V
    at q8_0 because long-context testing gets the VRAM win without the
    quality regressions seen from more aggressive cache choices.
    """
    return "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0"


def _reasoning_disabled_args(model_path: str) -> str:
    return (
        f"{_kv_cache_args(model_path)} --reasoning off --reasoning-budget 0 --reasoning-format none"
    )


def _apply_rpc_split_default(config: dict[str, Any], split: str) -> None:
    if str(config.get("runtime_mode") or "local").strip().lower() != "rpc":
        return
    if str(config.get("rpc_tensor_split") or "").strip():
        return
    config["rpc_tensor_split"] = split


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


def _preset_label(model_path: str) -> str:
    parts: list[str] = []
    name = _name(model_path)
    if _is_qwen36(model_path):
        parts.append("Qwen3.6")
    elif "qwen" in name:
        parts.append("Qwen")
    elif "gemma" in name:
        parts.append("Gemma")
    elif "glm" in name:
        parts.append("GLM")

    model_size = detect_model_size_billions(model_path)
    if model_size is not None:
        parts.append(f"{model_size:g}B")
    else:
        parts.append(Path(model_path).stem[:28])

    if is_moe_model(model_path):
        parts.append("MoE")
    if is_reap_model(model_path):
        parts.append("REAP")
    parts.append("CUDA")
    return " ".join(parts)


def list_candidate_models() -> list[str]:
    candidates: list[str] = []
    for root in model_roots():
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            if _is_excluded_model(path.name):
                continue
            candidates.append(str(path))
    return sorted(set(candidates), key=model_sort_key, reverse=True)


def model_sort_key(model_path: str) -> tuple[int, int, int, float, int, str]:
    name = _name(model_path)
    size_score = detect_model_size_billions(model_path) or 0.0

    qwen36_score = 1 if _is_qwen36(model_path) else 0
    a3b_score = 1 if "a3b" in name else 0
    non_reap_score = 0 if is_reap_model(model_path) else 1
    quant_score = (
        2
        if "ud-iq3_s" in name
        else 1
        if "q3_k_m" in name or "q4_k_m" in name or "q4_k_xl" in name
        else 0
    )
    return (qwen36_score, a3b_score, non_reap_score, size_score, quant_score, name)


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
        vram_info = estimate_vram_gib(model_path)
        validation = _validation_metadata(model_path)
        models.append(
            {
                "path": model_path,
                "name": name,
                "size_bytes": size_bytes,
                "size_gib": round(size_bytes / (1024**3), 2),
                "is_reap": "reap" in lower or "prune" in lower or "pruned" in lower,
                "is_a3b": "a3b" in lower,
                "is_moe": is_moe_model(model_path),
                "model_size_b": vram_info["model_size_b"],
                "vram": vram_info,
                "sort_key": model_sort_key(model_path),
                **validation,
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
    custom_args: str | None = None,
) -> dict[str, Any]:
    profile = _rk3588_profile()
    profile.update(
        {
            "ctx_size": ctx_size,
            "parallel": 1,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "max_tokens": max_tokens,
            "custom_args": custom_args or _platform_custom_args(""),
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

    if _is_qwen36_moe_35b(model_path):
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=131072,
            batch_size=512,
            ubatch_size=64,
            max_tokens=2048,
            custom_args=f"{_kv_cache_args(model_path)} --no-mmap --reasoning off --cache-ram 0",
        )

    if _is_qwen36_dense_27b(model_path):
        return _apply_cpu_model_profile(
            tuned,
            ctx_size=65536,
            batch_size=512,
            ubatch_size=64,
            max_tokens=2048,
            custom_args=f"{_kv_cache_args(model_path)} --no-mmap --reasoning off --cache-ram 0",
        )

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
            custom_args=_reasoning_disabled_args(model_path),
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


def _cuda_profile_for_model(config: dict[str, Any], _lower: str, model_path: str) -> dict[str, Any]:
    """Apply CUDA-optimized profile with automatic GPU layer estimation."""
    model_size = detect_model_size_billions(model_path)
    gpu_layers = suggest_gpu_layers(model_path)
    kv_args = _kv_cache_args(model_path)

    if _is_qwen36_moe_35b(model_path):
        config.update(
            {
                "gpu_layers": 999,
                "ctx_size": 131072,
                "parallel": 1,
                "batch_size": 512,
                "ubatch_size": 64,
                "max_tokens": 2048,
                "custom_args": f"{kv_args} --no-mmap --reasoning off --cache-ram 0",
            }
        )
        _apply_rpc_split_default(config, "34,66")
        return config

    if _is_qwen36_dense_27b(model_path):
        config.update(
            {
                "gpu_layers": 999,
                "ctx_size": 65536,
                "parallel": 1,
                "batch_size": 512,
                "ubatch_size": 64,
                "max_tokens": 2048,
                "custom_args": f"{kv_args} --no-mmap --reasoning off --cache-ram 0",
            }
        )
        _apply_rpc_split_default(config, "34,66")
        return config

    if _is_qwen_27b(model_path) or (model_size is not None and model_size >= 27):
        # 27B models need partial offload on 12GB VRAM
        # Use smaller context to save VRAM for weights
        ctx = 4096 if gpu_layers < 99 else 32768
        config.update(
            {
                "gpu_layers": gpu_layers,
                "ctx_size": ctx,
                "parallel": 1,
                "batch_size": 512,
                "ubatch_size": 128,
                "max_tokens": 2048,
                "custom_args": kv_args,
            }
        )
        return config

    if model_size is not None and model_size >= 32:
        ctx = 8192 if gpu_layers < 99 else 16384
        config.update(
            {
                "gpu_layers": gpu_layers,
                "ctx_size": ctx,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": kv_args,
            }
        )
        return config

    if model_size is not None and model_size >= 13:
        config.update(
            {
                "gpu_layers": gpu_layers,
                "ctx_size": 32768,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": kv_args,
            }
        )
        return config

    if model_size is not None and model_size >= 6:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 32768,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 2048,
                "custom_args": kv_args,
            }
        )
        return config

    if model_size is not None and model_size < 6:
        config.update(
            {
                "gpu_layers": 99,
                "ctx_size": 65536,
                "parallel": 4,
                "batch_size": 512,
                "ubatch_size": 256,
                "max_tokens": 4096,
                "custom_args": kv_args,
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


def build_model_presets(
    defaults: dict[str, Any], candidates: list[str] | None = None
) -> list[dict[str, Any]]:
    available = list(candidates or list_candidate_models())
    presets: list[dict[str, Any]] = []

    cuda_validated: tuple[tuple[Callable[[str], bool], str, str], ...] = (
        (
            lambda p: _is_qwen36_moe_35b(p) and "ud-iq3_s" in _name(p),
            "Qwen3.6 MoE 35B 128K",
            "Best measured profile: 128K Q8 KV, Flash Attention, 39.6 tok/s, quality tests passed.",
        ),
        (
            _is_qwen36_dense_27b,
            "Qwen3.6 Dense 27B 64K",
            "Validated dense fallback: 64K Q8 KV, 18.5 tok/s, quality tests passed.",
        ),
        (
            lambda p: _is_qwen35_size(p, 9),
            "Qwen3.5 9B CUDA",
            "Validated smaller quality tier; practical on 12GB-class CUDA.",
        ),
        (
            lambda p: _is_qwen35_size(p, 4),
            "Qwen3.5 4B CUDA",
            "Validated working fast tier and known WebUI baseline.",
        ),
    )

    if GPU_BACKEND == "cuda":
        seen_paths: set[str] = set()

        def append_preset(
            match: str, label: str | None = None, description: str | None = None
        ) -> None:
            if match in seen_paths:
                return
            seen_paths.add(match)
            vram = estimate_vram_gib(match)
            gpu_layers = suggest_gpu_layers(match)
            fits_note = " (partial GPU offload)" if gpu_layers < 99 else ""
            presets.append(
                {
                    "id": f"cuda-{Path(match).stem}",
                    "label": label or _preset_label(match),
                    "description": description
                    or f"Benchmark-validated CUDA profile. Est. {vram['min_vram_gib']} GiB VRAM{fits_note}.",
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
                        match,
                    ),
                }
            )

        for finder, label_override, description_override in cuda_validated:
            match = _find_matching_model(available, finder)
            if match:
                append_preset(match, label_override, description_override)

        return presets

    day_model = _find_matching_model(available, _is_day_qwen)
    night_model = _find_matching_model(available, _is_glm_flash_reap)

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
