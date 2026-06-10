"""HuggingFace model discovery for hardware-constrained environments.

Queries the HF API to find GGUF models that fit the current hardware,
ranks them by a composite score (MoE preference, size tier, quant quality,
popularity), and exposes results via CLI and API.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HF_API_BASE = "https://huggingface.co/api"
CACHE_TTL_SECONDS = 1800  # 30 minutes

# Quant preference ranking (higher = better for constrained hardware)
_QUANT_SCORES: dict[str, int] = {
    "Q4_K_M": 10,
    "Q4_K_S": 8,
    "Q4_0": 7,
    "Q5_K_M": 9,
    "Q5_K_S": 8,
    "Q5_0": 7,
    "Q3_K_M": 5,
    "Q3_K_S": 4,
    "Q3_K_L": 6,
    "Q8_0": 6,
    "Q6_K": 7,
    "Q2_K": 3,
    "IQ4_XS": 7,
    "IQ3_M": 5,
    "IQ3_S": 4,
}

# Size tier scoring: larger models are smarter, but must fit in RAM
_SIZE_TIER_SCORES: list[tuple[str, int]] = [
    ("120b", 20),
    ("72b", 18),
    ("70b", 18),
    ("35b", 16),
    ("34b", 16),
    ("32b", 16),
    ("28b", 15),
    ("27b", 15),
    ("26b", 15),
    ("24b", 14),
    ("23b", 14),
    ("19b", 13),
    ("18b", 13),
    ("16b", 12),
    ("14b", 11),
    ("9b", 10),
    ("8b", 9),
    ("7b", 8),
    ("4b", 6),
    ("3b", 5),
    ("2b", 4),
    ("1.7b", 3),
    ("0.8b", 2),
]


@dataclass
class HardwareProfile:
    """Hardware constraints for model filtering."""

    total_ram_gb: float
    max_model_gb: float
    gpu_backend: str = "cpu"
    is_arm: bool = True

    @classmethod
    def from_system(cls, ram_gb: float | None = None) -> HardwareProfile:
        """Detect hardware profile from the running system."""
        if ram_gb is None:
            ram_gb = _detect_total_ram_gb()
        # Leave ~7 GB for OS + context + other processes
        max_model_gb = max(ram_gb - 7.0, 4.0)
        machine = _get_machine()
        is_arm = machine.startswith("aarch64") or machine.startswith("arm")
        gpu_backend = _detect_gpu_backend()
        return cls(
            total_ram_gb=ram_gb,
            max_model_gb=max_model_gb,
            gpu_backend=gpu_backend,
            is_arm=is_arm,
        )


@dataclass
class GgufFile:
    """A single GGUF file in a HF repo."""

    filename: str
    size_bytes: int
    size_gb: float = field(init=False)
    quant_label: str = field(init=False)
    quant_score: int = field(init=False)

    def __post_init__(self) -> None:
        self.size_gb = round(self.size_bytes / (1024**3), 2)
        self.quant_label = _extract_quant(self.filename)
        self.quant_score = _QUANT_SCORES.get(self.quant_label, 0)


@dataclass
class DiscoveredModel:
    """A model discovered on HuggingFace with ranking metadata."""

    repo_id: str
    model_name: str
    is_moe: bool
    gguf_files: list[GgufFile]
    fits_hardware: bool = False
    rank_score: int = 0
    downloads: int = 0
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str = ""
    best_gguf: GgufFile | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "model_name": self.model_name,
            "is_moe": self.is_moe,
            "gguf_files": [
                {"filename": f.filename, "size_gb": f.size_gb, "quant": f.quant_label}
                for f in self.gguf_files
            ],
            "best_gguf": (
                {
                    "filename": self.best_gguf.filename,
                    "size_gb": self.best_gguf.size_gb,
                    "quant": self.best_gguf.quant_label,
                }
                if self.best_gguf
                else None
            ),
            "fits_hardware": self.fits_hardware,
            "rank_score": self.rank_score,
            "downloads": self.downloads,
            "tags": self.tags,
            "pipeline_tag": self.pipeline_tag,
        }


def _detect_total_ram_gb() -> float:
    """Read total RAM from /proc/meminfo."""
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / (1024**2), 2)
    except (OSError, ValueError, IndexError):
        pass
    return 16.0  # safe default


def _get_machine() -> str:
    import platform

    return platform.machine().lower()


def _detect_gpu_backend() -> str:
    try:
        from .settings import GPU_BACKEND

        return GPU_BACKEND
    except ImportError:
        return "cpu"


def _extract_quant(filename: str) -> str:
    """Extract the quantization label from a GGUF filename."""
    upper = filename.upper()
    for label in _QUANT_SCORES:
        if label in upper:
            return label
    return ""


def _extract_size_tier(name: str) -> int:
    """Extract model size tier score from name."""
    lower = name.lower()
    for hint, score in _SIZE_TIER_SCORES:
        if hint in lower:
            return score
    return 1


def _is_moe(name: str) -> bool:
    """Check if a model name suggests MoE architecture."""
    lower = name.lower()
    return any(tag in lower for tag in ("a3b", "a4b", "a6b", "moe", "-moe-"))


def _hf_api_request(url: str, timeout: int = 30) -> dict[str, Any] | list[Any] | None:
    """Make a GET request to the HF API. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: dict[str, Any] | list[Any] = json.loads(resp.read().decode("utf-8"))
            return data
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def _cached_results(key: str) -> Any | None:
    """Retrieve cached results if still valid."""
    entry = _cache.get(key)
    if entry is None:
        return None
    if time.monotonic() - entry["ts"] > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return entry["data"]


def _store_cache(key: str, data: Any) -> None:
    _cache[key] = {"ts": time.monotonic(), "data": data}


_cache: dict[str, dict[str, Any]] = {}


def search_models(
    query: str = "",
    sort: str = "",
    limit: int = 50,
    author: str = "",
) -> list[dict[str, Any]]:
    """Search HF models with GGUF tag. Returns raw API response items."""
    cache_key = f"search:{query}:{sort}:{limit}:{author}"
    cached = _cached_results(cache_key)
    if cached is not None:
        return list(cached)

    params = [f"limit={limit}", "filter=gguf"]
    if query:
        params.append(f"search={query}")
    if sort:
        params.append(f"sort={sort}")
    if author:
        params.append(f"author={author}")

    url = f"{HF_API_BASE}/models?{'&'.join(params)}"
    result = _hf_api_request(url)

    if result is None or not isinstance(result, list):
        return []

    _store_cache(cache_key, result)
    return list(result)


def get_gguf_files(repo_id: str) -> list[GgufFile]:
    """List GGUF files with sizes for a given HF repo."""
    cache_key = f"files:{repo_id}"
    cached = _cached_results(cache_key)
    if cached is not None:
        return list(cached)

    url = f"{HF_API_BASE}/models/{repo_id}/tree/main"
    data = _hf_api_request(url)
    if data is None or not isinstance(data, list):
        return []

    files: list[GgufFile] = []
    for entry in data:
        path = entry.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        size = entry.get("size", 0)
        files.append(GgufFile(filename=path, size_bytes=size))

    _store_cache(cache_key, files)
    return files


def rank_models(
    models: list[dict[str, Any]],
    hardware: HardwareProfile,
    arch_pref: str = "any",
) -> list[DiscoveredModel]:
    """Convert raw HF search results into ranked DiscoveredModel objects."""
    discovered: list[DiscoveredModel] = []

    for model_data in models:
        repo_id = model_data.get("id", "")
        if not repo_id:
            continue

        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        is_moe = _is_moe(model_name)
        downloads = model_data.get("downloads", 0)
        tags = model_data.get("tags", [])
        pipeline_tag = model_data.get("pipeline_tag", "")

        # Filter by architecture preference
        if arch_pref == "moe" and not is_moe:
            continue
        if arch_pref == "dense" and is_moe:
            continue

        gguf_files = get_gguf_files(repo_id)
        if not gguf_files:
            continue

        # Find the best-fitting GGUF file for this hardware
        fitting_files = [f for f in gguf_files if f.size_gb <= hardware.max_model_gb]
        fits = len(fitting_files) > 0
        best = max(fitting_files, key=lambda f: f.quant_score) if fitting_files else None

        # If nothing fits, pick the smallest anyway (user might want to know about it)
        if best is None and gguf_files:
            best = min(gguf_files, key=lambda f: f.size_gb)

        # Compute rank score
        score = 0
        score += _extract_size_tier(model_name)
        score += 10 if is_moe else 0
        score += best.quant_score if best else 0
        score += 5 if fits else 0
        # Popularity: +1 per 10k downloads, capped at 10
        score += min(downloads // 10_000, 10)

        discovered.append(
            DiscoveredModel(
                repo_id=repo_id,
                model_name=model_name,
                is_moe=is_moe,
                gguf_files=gguf_files,
                fits_hardware=fits,
                rank_score=score,
                downloads=downloads,
                tags=tags,
                pipeline_tag=pipeline_tag,
                best_gguf=best,
            )
        )

    discovered.sort(key=lambda m: m.rank_score, reverse=True)
    return discovered


def recommend_models(
    hardware: HardwareProfile | None = None,
    limit: int = 10,
    query: str = "",
    arch_pref: str = "any",
) -> list[DiscoveredModel]:
    """End-to-end: search HF, filter by hardware, rank, return top N."""
    if hardware is None:
        hardware = HardwareProfile.from_system()

    raw = search_models(query=query, limit=100)
    ranked = rank_models(raw, hardware, arch_pref=arch_pref)

    # Prefer models that fit hardware first
    fitting = [m for m in ranked if m.fits_hardware]
    not_fitting = [m for m in ranked if not m.fits_hardware]
    ordered = fitting + not_fitting

    return ordered[:limit]


def cli_main() -> None:
    """CLI entry point for hf-discover."""
    parser = argparse.ArgumentParser(
        description="Discover GGUF models from HuggingFace that fit your hardware.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=None,
        help="Override max model size in GB (default: auto-detect from RAM)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of models to show (default: 15)",
    )
    parser.add_argument(
        "--arch",
        choices=["any", "moe", "dense"],
        default="any",
        help="Architecture preference (default: any)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Search query to filter models",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON instead of formatted table",
    )
    args = parser.parse_args()

    hardware = HardwareProfile.from_system()
    if args.max_gb is not None:
        hardware.max_model_gb = args.max_gb

    if args.output_json:
        results = recommend_models(hardware, limit=args.limit, query=args.query, arch_pref=args.arch)
        print(json.dumps([m.to_dict() for m in results], indent=2))
        return

    results = recommend_models(hardware, limit=args.limit, query=args.query, arch_pref=args.arch)

    if not results:
        print("No models found. Check your network connection or try a different query.")
        return

    # Header
    hw_label = f"{hardware.total_ram_gb:.1f}GB RAM" if hardware.total_ram_gb else "unknown RAM"
    print(f"\nHardware: {hw_label}, max model size: {hardware.max_model_gb:.1f}GB, backend: {hardware.gpu_backend}")
    print(f"Showing top {len(results)} models (arch: {args.arch})\n")

    # Table
    print(f"{'#':<3} {'Score':<6} {'Fits':<5} {'MoE':<4} {'Best GGUF Size':<16} {'Quant':<8} {'Repo'}")
    print("-" * 90)
    for i, model in enumerate(results, 1):
        fits_marker = "YES" if model.fits_hardware else "no"
        moe_marker = "YES" if model.is_moe else ""
        size_str = f"{model.best_gguf.size_gb:.1f}GB" if model.best_gguf else "?"
        quant_str = model.best_gguf.quant_label if model.best_gguf else "?"
        print(f"{i:<3} {model.rank_score:<6} {fits_marker:<5} {moe_marker:<4} {size_str:<16} {quant_str:<8} {model.repo_id}")

    # Detail for top 3
    print(f"\n--- Top {min(3, len(results))} Details ---\n")
    for model in results[:3]:
        print(f"  {model.repo_id}")
        print(f"    Downloads: {model.downloads:,} | MoE: {model.is_moe} | Tags: {', '.join(model.tags[:5])}")
        if model.gguf_files:
            print(f"    GGUF files ({len(model.gguf_files)}):")
            for f in sorted(model.gguf_files, key=lambda x: x.size_gb)[:5]:
                fits_flag = "OK" if f.size_gb <= hardware.max_model_gb else "TOO BIG"
                print(f"      {f.filename} ({f.size_gb:.1f}GB) [{fits_flag}]")
        print()
