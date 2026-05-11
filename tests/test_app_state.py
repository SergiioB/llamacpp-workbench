from __future__ import annotations

from llama_webui.app_state import _normalize_q8_kv_args, _normalize_rpc_config


def test_normalize_q8_kv_args_replaces_old_cache_flags():
    normalized = _normalize_q8_kv_args(
        "--cache-type-k f16 --cache-type-v q4_0 --flash-attn off --threads 8"
    )

    assert "--threads 8" in normalized
    assert "--cache-type-k q8_0" in normalized
    assert "--cache-type-v q8_0" in normalized
    assert "--flash-attn on" in normalized
    assert "f16" not in normalized
    assert "q4_0" not in normalized


def test_normalize_rpc_config_clears_legacy_local_defaults():
    config = {
        "runtime_mode": "local",
        "rpc_host": "192.0.2.60",
        "rpc_tensor_split": "34,66",
    }

    _normalize_rpc_config(config)

    assert config["rpc_host"] == ""
    assert config["rpc_tensor_split"] == ""
