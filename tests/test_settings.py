from __future__ import annotations

import os
from pathlib import Path

from llama_webui import settings


def test_split_paths_expands_and_skips_empty(tmp_path):
    home_models = Path("~").expanduser() / "models"
    value = f"{tmp_path}{os.pathsep} {os.pathsep}~/models"

    paths = settings._split_paths(value)

    assert paths == [tmp_path, home_models]


def test_default_download_dir_prefers_existing_model_root(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setenv("LLAMA_WEBUI_MODEL_DIRS", f"{first}{os.pathsep}{second}")

    assert settings.default_download_dir() == second


def test_resolve_binary_uses_env_override(monkeypatch, tmp_path):
    binary = tmp_path / "custom-llama-server"
    monkeypatch.setenv("LLAMA_WEBUI_LLAMA_SERVER", str(binary))

    assert settings.resolve_llama_server_binary() == str(binary)
