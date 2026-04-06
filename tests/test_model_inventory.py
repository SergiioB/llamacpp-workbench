from __future__ import annotations

from llama_webui import model_inventory


def test_select_preferred_model_prefers_reap_a3b_candidate():
    candidates = [
        "/models/Qwen3.5-4B-Q4_K_M.gguf",
        "/models/GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf",
        "/models/Qwen3.5-27B-Q4_K_M.gguf",
    ]

    preferred = model_inventory.select_preferred_model(candidates)

    assert preferred == "/models/GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf"


def test_normalize_model_path_replaces_missing_legacy_name(monkeypatch, tmp_path):
    preferred = str(tmp_path / "preferred.gguf")
    candidates = [preferred]
    monkeypatch.setattr(
        model_inventory,
        "LEGACY_DEFAULT_MODEL_FILENAMES",
        {"qwen3.5-4b-q4_k_m.gguf"},
    )

    normalized = model_inventory.normalize_model_path(
        str(tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"),
        candidates,
    )

    assert normalized == preferred


def test_apply_model_profile_sets_reap_cpu_tuning(monkeypatch):
    monkeypatch.setattr(model_inventory, "GPU_BACKEND", "cpu")
    config = {"gpu_layers": 0, "ctx_size": 2048, "threads": 8, "parallel": 1}

    tuned = model_inventory.apply_model_profile(
        config,
        "/models/GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf",
    )

    assert tuned["ctx_size"] == 202752
    assert tuned["threads"] == 4
    assert "--reasoning-budget 0" in tuned["custom_args"]
