"""Launch preflight checks for llama-server.

Runs before every start attempt to catch common failure modes:
- Local CUDA VRAM too low for the selected model + tensor split
- Remote RPC endpoint unreachable or protocol-incompatible
- llama-server log OOM parsing with suggested fix
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

from .settings import GPU_BACKEND


def check_local_vram() -> dict[str, Any]:
    """Query local NVIDIA GPU free VRAM via nvidia-smi.

    Returns free_mib=0 and available=False on non-CUDA systems or when
    nvidia-smi is not installed.
    """
    if GPU_BACKEND != "cuda":
        return {"available": False, "reason": "no_cuda", "free_mib": 0, "total_mib": 0}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"available": False, "reason": "nvidia_smi_failed", "free_mib": 0, "total_mib": 0}

        # Parse first GPU line: "5978, 12288"
        lines = result.stdout.strip().splitlines()
        if not lines:
            return {"available": False, "reason": "nvidia_smi_empty", "free_mib": 0, "total_mib": 0}

        parts = lines[0].split(",")
        if len(parts) < 2:
            return {"available": False, "reason": "nvidia_smi_parse_error", "free_mib": 0, "total_mib": 0}

        free_mib = int(parts[0].strip())
        total_mib = int(parts[1].strip())
        return {
            "available": True,
            "reason": None,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "free_gib": round(free_mib / 1024, 2),
            "total_gib": round(total_mib / 1024, 2),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        return {"available": False, "reason": "nvidia_smi_unavailable", "free_mib": 0, "total_mib": 0}


def _estimate_local_allocation_gib(
    model_path: str,
    rpc_tensor_split: str,
    runtime_mode: str,
) -> float | None:
    """Estimate how much VRAM the coordinator needs locally.

    For RPC mode with a tensor split like "34,66", the local GPU gets
    roughly 34% of the model weights.  For local mode the local GPU gets
    100%.

    Returns None if the estimate cannot be computed.
    """
    path = Path(model_path)
    if not path.exists():
        return None
    file_gib = path.stat().st_size / (1024 ** 3)

    if str(runtime_mode).strip().lower() != "rpc":
        # Full model on local GPU
        return file_gib

    split_str = str(rpc_tensor_split or "").strip()
    if not split_str:
        # No split info — assume full local
        return file_gib

    try:
        parts = [float(p.strip()) for p in split_str.split(",")]
    except (ValueError, IndexError):
        return file_gib

    if not parts:
        return file_gib

    total = sum(parts)
    if total <= 0:
        return file_gib

    local_ratio = parts[0] / total
    return file_gib * local_ratio


def check_vram_fit(
    model_path: str,
    runtime_mode: str,
    rpc_tensor_split: str,
    headroom_gib: float = 1.5,
) -> dict[str, Any]:
    """Check if the model fits in local VRAM.

    Returns a dict with:
      - fits: bool
      - local_vram: result from check_local_vram()
      - estimated_local_gib: estimated local allocation
      - headroom_gib: VRAM reserved for CUDA/runtime overhead
      - message: human-readable summary
      - suggestion: actionable fix (when fits=False)
    """
    vram = check_local_vram()
    estimated = _estimate_local_allocation_gib(model_path, rpc_tensor_split, runtime_mode)

    result: dict[str, Any] = {
        "local_vram": vram,
        "estimated_local_gib": round(estimated, 2) if estimated is not None else None,
        "headroom_gib": headroom_gib,
        "fits": None,
        "message": "",
        "suggestion": "",
    }

    if not vram["available"]:
        result["fits"] = None  # Unknown — cannot check
        result["message"] = "VRAM check skipped (no NVIDIA GPU or nvidia-smi unavailable)"
        result["suggestion"] = ""
        return result

    if estimated is None:
        result["fits"] = None
        result["message"] = "Cannot estimate local allocation (model file not found)"
        result["suggestion"] = "Select a valid model file"
        return result

    free_gib = vram["free_gib"]
    needed_gib = estimated + headroom_gib
    result["needed_gib"] = round(needed_gib, 2)

    if free_gib >= needed_gib:
        result["fits"] = True
        result["message"] = (
            f"VRAM OK: {free_gib} GiB free, ~{needed_gib} GiB needed "
            f"({estimated} GiB model + {headroom_gib} GiB headroom)"
        )
    else:
        deficit = round(needed_gib - free_gib, 2)
        result["fits"] = False
        result["message"] = (
            f"VRAM too low: {free_gib} GiB free, ~{needed_gib} GiB needed "
            f"({estimated} GiB model + {headroom_gib} GiB headroom). "
            f"Short by {deficit} GiB."
        )
        result["suggestion"] = (
            f"Close GPU-heavy apps (browsers, Discord, Steam, game overlays) "
            f"to free ~{deficit} GiB, or switch to a smaller model/profile."
        )

    return result


def check_rpc_protocol(host: str, port: int, timeout: float = 3.0) -> dict[str, Any]:
    """Check whether the RPC endpoint accepts TCP connections.

    This is intentionally a passive TCP check. llama.cpp's RPC server does not
    send a greeting before the client speaks, and readiness polling must not
    read from the RPC socket while a model is loading.

    Returns:
      - reachable: bool
      - protocol_ok: bool | None (None = couldn't verify)
      - error: str (when not reachable)
      - message: human-readable summary
    """
    result: dict[str, Any] = {
        "reachable": False,
        "protocol_ok": None,
        "error": "",
        "message": "",
    }

    if not host or port <= 0:
        result["error"] = "RPC host and port are required"
        result["message"] = result["error"]
        return result

    try:
        with socket.create_connection((host, port), timeout=timeout):
            result["reachable"] = True
            result["message"] = f"RPC endpoint {host}:{port} reachable"
            return result
    except OSError as exc:
        result["error"] = str(exc)
        result["message"] = f"RPC endpoint {host}:{port} unreachable: {exc}"

    return result


_OOM_PATTERN = re.compile(r"cudaMalloc failed: out of memory", re.IGNORECASE)
_ALLOC_PATTERN = re.compile(r"allocating\s+([\d.]+)\s+MiB on device", re.IGNORECASE)
_MODEL_LOAD_ERROR = re.compile(r"failed to load model", re.IGNORECASE)
_PROTOCOL_MISMATCH = re.compile(r"HELLO request size mismatch|protocol version", re.IGNORECASE)


def parse_log_errors(log_text: str) -> list[dict[str, str]]:
    """Parse llama-server log for known error patterns and return diagnoses."""
    diagnoses: list[dict[str, str]] = []
    if not log_text:
        return diagnoses

    # CUDA OOM
    if _OOM_PATTERN.search(log_text):
        alloc_match = _ALLOC_PATTERN.search(log_text)
        alloc_gib = f"{float(alloc_match.group(1)) / 1024:.1f} GiB" if alloc_match else "unknown amount"
        diagnoses.append({
            "type": "cuda_oom",
            "severity": "critical",
            "title": "CUDA Out of Memory",
            "detail": f"llama-server tried to allocate {alloc_gib} on the local GPU and failed.",
            "suggestion": (
                "Close GPU-heavy apps (browsers, Discord, Steam, game overlays, Edge webviews) "
                "to free VRAM, then try again. Alternatively, use a smaller model or a different "
                "tensor split ratio."
            ),
        })

    # Protocol mismatch
    if _PROTOCOL_MISMATCH.search(log_text):
        diagnoses.append({
            "type": "protocol_mismatch",
            "severity": "critical",
            "title": "RPC Protocol Mismatch",
            "detail": "The RPC worker runs a different llama.cpp version than the coordinator.",
            "suggestion": (
                "Update the RPC worker to the same llama.cpp build as the coordinator. "
                "Download a matching binary from the same release."
            ),
        })

    # Generic model load failure (when no more specific pattern matched)
    if _MODEL_LOAD_ERROR.search(log_text) and not _OOM_PATTERN.search(log_text) and not _PROTOCOL_MISMATCH.search(log_text):
        diagnoses.append({
            "type": "model_load_error",
            "severity": "critical",
            "title": "Model Load Failed",
            "detail": "llama-server failed to load the model. Check the log for details.",
            "suggestion": "Check the full log for details. Common causes: corrupt model file, incompatible GGUF version, or insufficient resources.",
        })

    return diagnoses


def run_preflight(
    config: dict[str, Any],
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Run all preflight checks and return a combined result.

    Returns:
      - ready: bool — True if all critical checks pass
      - checks: dict of individual check results
      - blocking_issues: list of human-readable blocking issues
      - warnings: list of human-readable warnings
      - log_diagnoses: list of parsed log error diagnoses
    """
    runtime_mode = str(config.get("runtime_mode") or "local").strip().lower()
    rpc_enabled = runtime_mode == "rpc"
    model_path = str(config.get("model_path") or "")
    rpc_host = str(config.get("rpc_host") or "").strip()
    rpc_port = int(config.get("rpc_port") or 0)
    rpc_tensor_split = str(config.get("rpc_tensor_split") or "").strip()

    # 1. Local VRAM check
    vram_check = check_vram_fit(model_path, runtime_mode, rpc_tensor_split)

    # 2. RPC check (only when RPC mode)
    rpc_check: dict[str, Any] | None = None
    if rpc_enabled:
        rpc_check = check_rpc_protocol(rpc_host, rpc_port)
        # Also validate tensor split is set
        if not rpc_tensor_split:
            rpc_check["tensor_split_ok"] = False
            rpc_check["message"] += " (tensor split is required)"
        else:
            rpc_check["tensor_split_ok"] = True

    # 3. Log error parsing
    log_diagnoses: list[dict[str, str]] = []
    if log_path and log_path.exists():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            log_diagnoses = parse_log_errors(log_text)
        except OSError:
            pass

    # 4. Binary exists check
    binary_path = str(config.get("llama_binary") or "")
    binary_exists = os.path.isfile(binary_path) if binary_path else False

    # 5. Model file exists check
    model_exists = os.path.isfile(model_path) if model_path else False

    # Aggregate
    blocking_issues: list[str] = []
    warnings: list[str] = []

    if not binary_exists:
        blocking_issues.append(f"llama-server binary not found: {binary_path}")

    if not model_exists:
        blocking_issues.append(f"Model file not found: {model_path}")

    if vram_check["fits"] is False:
        blocking_issues.append(vram_check["message"])
        if vram_check.get("suggestion"):
            warnings.append(vram_check["suggestion"])

    if rpc_enabled and rpc_check:
        if not rpc_check["reachable"]:
            blocking_issues.append(rpc_check["message"])
        if rpc_check.get("tensor_split_ok") is False:
            blocking_issues.append("RPC mode requires a tensor split (e.g., 34,66)")
        if rpc_check.get("protocol_ok") is False:
            warnings.append("RPC protocol version may not match the coordinator")

    for diag in log_diagnoses:
        if diag.get("severity") == "critical":
            warnings.append(f"{diag['title']}: {diag['suggestion']}")

    ready = len(blocking_issues) == 0

    checks: dict[str, Any] = {
        "vram": vram_check,
        "binary_exists": binary_exists,
        "model_exists": model_exists,
    }
    if rpc_check is not None:
        checks["rpc"] = rpc_check

    return {
        "ready": ready,
        "checks": checks,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "log_diagnoses": log_diagnoses,
    }
