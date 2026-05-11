from __future__ import annotations

from llama_webui.llama_manager import LlamaServerManager


def _base_config(tmp_path):
    return {
        "llama_binary": str(tmp_path / "llama-server"),
        "llama_host": "127.0.0.1",
        "llama_port": 8085,
        "model_path": str(tmp_path / "model.gguf"),
        "ctx_size": 131072,
        "threads": 8,
        "parallel": 1,
        "gpu_layers": 99,
        "batch_size": 512,
        "ubatch_size": 64,
        "gpu_backend": "auto",
        "custom_args": "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0",
    }


def test_sanitize_visible_content_removes_think_blocks(tmp_path):
    manager = LlamaServerManager(tmp_path / "llama.log")

    cleaned = manager._sanitize_visible_content("<think>hidden</think>\nVisible answer")

    assert cleaned == "Visible answer"


def test_sanitize_visible_content_keeps_plain_text(tmp_path):
    manager = LlamaServerManager(tmp_path / "llama.log")

    cleaned = manager._sanitize_visible_content("  Plain answer  ")

    assert cleaned == "Plain answer"


def test_managed_pid_removes_stale_pid_file(monkeypatch, tmp_path):
    log_path = tmp_path / "llama.log"
    pid_path = log_path.with_name("llama-server.pid")
    pid_path.write_text("4321\n", encoding="utf-8")
    manager = LlamaServerManager(log_path)

    def fake_kill(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("llama_webui.llama_manager.os.kill", fake_kill)

    assert manager._managed_pid() is None
    assert not pid_path.exists()


def test_server_args_add_rpc_flags_before_custom_args(tmp_path):
    manager = LlamaServerManager(tmp_path / "llama.log")
    config = {
        **_base_config(tmp_path),
        "runtime_mode": "rpc",
        "rpc_host": "192.0.2.20",
        "rpc_port": 50052,
        "rpc_tensor_split": "40,60",
    }

    args = manager._server_args(config)

    assert "--rpc" in args
    assert args[args.index("--rpc") + 1] == "192.0.2.20:50052"
    assert args[args.index("--split-mode") + 1] == "layer"
    assert args[args.index("--tensor-split") + 1] == "40,60"
    assert args.index("--tensor-split") < args.index("--flash-attn")


def test_server_args_require_rpc_tensor_split(tmp_path):
    manager = LlamaServerManager(tmp_path / "llama.log")
    config = {
        **_base_config(tmp_path),
        "runtime_mode": "rpc",
        "rpc_host": "192.0.2.20",
        "rpc_port": 50052,
        "rpc_tensor_split": "",
    }

    try:
        manager._server_args(config)
    except RuntimeError as error:
        assert "tensor split" in str(error).lower()
    else:
        raise AssertionError("RPC start without tensor split should fail fast")


def test_rpc_preflight_success(monkeypatch, tmp_path):
    manager = LlamaServerManager(tmp_path / "llama.log")
    config = {
        **_base_config(tmp_path),
        "runtime_mode": "rpc",
        "rpc_host": "192.0.2.20",
        "rpc_port": 50052,
        "rpc_tensor_split": "",
    }

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_create_connection(address, timeout):
        assert address == ("192.0.2.20", 50052)
        assert timeout == 3.0
        return FakeSocket()

    monkeypatch.setattr(
        "llama_webui.llama_manager.socket.create_connection", fake_create_connection
    )

    assert manager.rpc_preflight(config) == {
        "enabled": True,
        "reachable": True,
        "endpoint": "192.0.2.20:50052",
    }


def test_rpc_preflight_reports_unreachable(monkeypatch, tmp_path):
    manager = LlamaServerManager(tmp_path / "llama.log")
    config = {
        **_base_config(tmp_path),
        "runtime_mode": "rpc",
        "rpc_host": "192.0.2.20",
        "rpc_port": 50052,
        "rpc_tensor_split": "",
    }

    def fake_create_connection(_address, timeout):
        assert timeout == 3.0
        raise OSError("connection refused")

    monkeypatch.setattr(
        "llama_webui.llama_manager.socket.create_connection", fake_create_connection
    )

    result = manager.rpc_preflight(config)
    assert result["enabled"] is True
    assert result["reachable"] is False
    assert result["endpoint"] == "192.0.2.20:50052"
    assert "connection refused" in result["error"]
