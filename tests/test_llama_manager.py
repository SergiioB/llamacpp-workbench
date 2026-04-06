from __future__ import annotations

from llama_webui.llama_manager import LlamaServerManager


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
