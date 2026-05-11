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


def test_qwen36_presets_include_moe_and_dense(monkeypatch):
    monkeypatch.setattr(model_inventory, "GPU_BACKEND", "cuda")
    candidates = [
        "/models/Qwen3.6-27B-Q3_K_M.gguf",
        "/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
        "/models/Qwen3.6-28B-REAP.Q4_K_M.gguf",
        "/models/Example-11B-Q4_K_M.gguf",
    ]

    presets = model_inventory.build_model_presets({"runtime_mode": "local"}, candidates)

    labels = [preset["label"] for preset in presets]
    assert "Qwen3.6 MoE 35B 128K" in labels
    assert "Qwen3.6 Dense 27B 64K" in labels
    assert "11B CUDA" not in labels

    moe_config = next(
        preset["config"] for preset in presets if preset["label"] == "Qwen3.6 MoE 35B 128K"
    )
    dense_config = next(
        preset["config"] for preset in presets if preset["label"] == "Qwen3.6 Dense 27B 64K"
    )
    assert moe_config["ctx_size"] == 131072
    assert dense_config["ctx_size"] == 65536
    assert "--cache-type-v q8_0" in moe_config["custom_args"]


def test_qwen36_rpc_profile_sets_validated_split(monkeypatch):
    monkeypatch.setattr(model_inventory, "GPU_BACKEND", "cuda")
    tuned = model_inventory.apply_model_profile(
        {"runtime_mode": "rpc", "rpc_tensor_split": ""},
        "/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
    )

    assert tuned["rpc_tensor_split"] == "34,66"


def test_validation_metadata_marks_tried_rejects():
    reap = model_inventory._validation_metadata("/models/Qwen3.6-28B-REAP.Q4_K_M.gguf")
    uncensored_q3kp = model_inventory._validation_metadata(
        "/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf"
    )
    uncensored_iq3m = model_inventory._validation_metadata(
        "/models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-IQ3_M.gguf"
    )

    assert reap["validation_status"] == "avoid"
    assert "rpc-tested" in reap["validation_note"].lower()
    assert "failed factual" in reap["validation_note"].lower()
    assert uncensored_q3kp["validation_status"] == "rpc_oom"
    assert "combined vram" in uncensored_q3kp["validation_note"].lower()
    assert uncensored_iq3m["validation_status"] == "rpc_constrained"
    assert "2/8 split" in uncensored_iq3m["validation_note"].lower()
