"""Tests for launch preflight checks."""

from __future__ import annotations

from pathlib import Path

from llama_webui.preflight import (
    check_local_vram,
    check_rpc_protocol,
    check_vram_fit,
    parse_log_errors,
    run_preflight,
)


def _base_config(tmp_path: Path) -> dict:
    """Create a minimal config dict for testing."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (100 * 1024 * 1024))  # 100 MiB dummy model
    return {
        "runtime_mode": "local",
        "model_path": str(model),
        "llama_binary": str(tmp_path / "llama-server"),
        "rpc_host": "",
        "rpc_port": 0,
        "rpc_tensor_split": "",
    }


def test_check_local_vram_returns_structure():
    """check_local_vram always returns a valid dict."""
    result = check_local_vram()
    assert "available" in result
    assert "free_mib" in result
    assert "total_mib" in result
    assert isinstance(result["available"], bool)


def test_check_vram_fit_model_not_found():
    """When model file doesn't exist, fits should be None."""
    result = check_vram_fit("/nonexistent/model.gguf", "local", "")
    assert result["fits"] is None
    assert "not found" in result["message"].lower() or "skipped" in result["message"].lower()


def test_check_vram_fit_local_mode(tmp_path):
    """Local mode should estimate 100% model on local GPU."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (100 * 1024 * 1024))

    result = check_vram_fit(str(model), "local", "")
    assert result["estimated_local_gib"] is not None
    # Should be ~0.1 GiB (100 MiB file)
    assert result["estimated_local_gib"] < 1.0


def test_check_vram_fit_rpc_split(tmp_path):
    """RPC mode with 34,66 split should allocate ~34% locally."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00" * (100 * 1024 * 1024))

    result = check_vram_fit(str(model), "rpc", "34,66")
    assert result["estimated_local_gib"] is not None
    # 34% of ~0.1 GiB = ~0.034 GiB
    assert result["estimated_local_gib"] < 0.05


def test_check_rpc_protocol_missing_host():
    """Empty host should return unreachable."""
    result = check_rpc_protocol("", 0)
    assert result["reachable"] is False
    assert "required" in result["error"].lower()


def test_check_rpc_protocol_unreachable(monkeypatch):
    """Connection refused should be reported as unreachable."""

    def fake_connect(_addr, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("llama_webui.preflight.socket.create_connection", fake_connect)
    result = check_rpc_protocol("10.0.0.99", 50052)
    assert result["reachable"] is False
    assert "connection refused" in result["error"]


def test_check_rpc_protocol_reachable(monkeypatch):
    """Successful connection should report reachable without reading from RPC."""

    class FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def recv(self, _n):
            raise AssertionError("preflight must not read from rpc-server")

    def fake_connect(_addr, **_kwargs):
        return FakeSock()

    monkeypatch.setattr("llama_webui.preflight.socket.create_connection", fake_connect)
    result = check_rpc_protocol("192.0.2.20", 50052)
    assert result["reachable"] is True
    assert result["protocol_ok"] is None


def test_parse_log_errors_oom():
    """CUDA OOM log should be detected."""
    log = """load_tensors: loading model tensors, this can take a while...
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8376.40 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 8783294720
llama_model_load: error loading model: unable to allocate CUDA0 buffer"""
    diagnoses = parse_log_errors(log)
    assert len(diagnoses) == 1
    assert diagnoses[0]["type"] == "cuda_oom"
    assert diagnoses[0]["severity"] == "critical"
    assert "8376" in diagnoses[0]["detail"] or "8.2 GiB" in diagnoses[0]["detail"]
    assert "close" in diagnoses[0]["suggestion"].lower()


def test_parse_log_errors_protocol_mismatch():
    """Protocol mismatch log should be detected."""
    log = "RPC error: HELLO request size mismatch (expected 42, got 36)"
    diagnoses = parse_log_errors(log)
    assert len(diagnoses) == 1
    assert diagnoses[0]["type"] == "protocol_mismatch"
    assert "update" in diagnoses[0]["suggestion"].lower()


def test_parse_log_errors_clean():
    """Clean log should produce no diagnoses."""
    log = "llama_server: listening on 127.0.0.1:8085\nModel loaded successfully."
    diagnoses = parse_log_errors(log)
    assert len(diagnoses) == 0


def test_parse_log_errors_empty():
    """Empty log should produce no diagnoses."""
    assert parse_log_errors("") == []
    assert parse_log_errors(None) == []


def test_run_preflight_local_missing_binary(tmp_path):
    """Missing binary should be a blocking issue."""
    config = _base_config(tmp_path)
    # Binary doesn't exist
    result = run_preflight(config)
    assert result["ready"] is False
    assert any("binary" in issue.lower() for issue in result["blocking_issues"])


def test_run_preflight_local_ready(tmp_path):
    """Valid local config should be ready (VRAM check skipped on non-CUDA)."""
    config = _base_config(tmp_path)
    # Create binary
    Path(config["llama_binary"]).write_bytes(b"\x00")
    # Model already exists from _base_config

    result = run_preflight(config)
    # On non-CUDA CI, VRAM is skipped, so should be ready
    # On CUDA, it depends on actual VRAM
    assert "binary_exists" in result["checks"]
    assert result["checks"]["binary_exists"] is True
    assert result["checks"]["model_exists"] is True


def test_run_preflight_rpc_missing_split(tmp_path):
    """RPC mode without tensor split should be blocking."""
    config = _base_config(tmp_path)
    Path(config["llama_binary"]).write_bytes(b"\x00")
    config["runtime_mode"] = "rpc"
    config["rpc_host"] = "192.0.2.20"
    config["rpc_port"] = 50052
    config["rpc_tensor_split"] = ""

    # Monkeypatch socket to simulate unreachable
    import llama_webui.preflight as pf

    original = pf.socket.create_connection

    def fake_connect(_addr, **_kwargs):
        raise OSError("connection refused")

    pf.socket.create_connection = fake_connect
    try:
        result = run_preflight(config)
    finally:
        pf.socket.create_connection = original

    assert result["ready"] is False
    assert any(
        "rpc" in issue.lower() or "unreachable" in issue.lower()
        for issue in result["blocking_issues"]
    )


def test_run_preflight_with_log_oom(tmp_path):
    """Previous OOM in log should appear in log_diagnoses."""
    config = _base_config(tmp_path)
    Path(config["llama_binary"]).write_bytes(b"\x00")

    log_path = tmp_path / "llama.log"
    log_path.write_text(
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8376.40 MiB on device 0: cudaMalloc failed: out of memory\n"
    )

    result = run_preflight(config, log_path=log_path)
    assert len(result["log_diagnoses"]) == 1
    assert result["log_diagnoses"][0]["type"] == "cuda_oom"
