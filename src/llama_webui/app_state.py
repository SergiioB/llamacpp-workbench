from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .model_inventory import list_candidate_models, normalize_model_path
from .settings import GPU_BACKEND, data_dir, resolve_llama_server_binary


def default_config() -> dict[str, Any]:
    candidates = list_candidate_models()
    detected_cpus = os.cpu_count() or 4
    cpu_threads = 4 if detected_cpus >= 8 else max(1, min(4, detected_cpus))
    gpu_layers = 99 if GPU_BACKEND == "cuda" else 0
    parallel = 4 if GPU_BACKEND == "cuda" else 1
    batch_size = 512 if GPU_BACKEND == "cuda" else 128
    ubatch_size = 128 if GPU_BACKEND == "cuda" else 32
    return {
        "bind_host": "0.0.0.0",
        "bind_port": 8095,
        "llama_host": "127.0.0.1",
        "llama_port": 8085,
        "llama_binary": resolve_llama_server_binary(),
        "model_path": normalize_model_path(None, candidates),
        "cpu_mask": "4-7",
        "ctx_size": 2048,
        "threads": cpu_threads,
        "gpu_layers": gpu_layers,
        "parallel": parallel,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": 512,
        "system_prompt": "",
        "custom_args": "--cache-type-k q8_0 --cache-type-v q4_0 --reasoning off --reasoning-budget 0 --reasoning-format none",
    }


DEFAULT_CONFIG: dict[str, Any] = default_config()


class AppState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "webui.db"
        self.log_path = self.data_dir / "llama-server.log"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS app_state (
                  key TEXT PRIMARY KEY,
                  value_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chats (
                  chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id INTEGER NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            if self._conn.execute("SELECT 1 FROM app_state WHERE key = 'config'").fetchone() is None:
                self.save_config(DEFAULT_CONFIG)
            self._conn.commit()

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT value_json FROM app_state WHERE key = 'config'").fetchone()
        if not row:
            return dict(DEFAULT_CONFIG)
        data = {**DEFAULT_CONFIG, **json.loads(row["value_json"])}
        data["model_path"] = normalize_model_path(str(data.get("model_path") or ""), list_candidate_models())
        return data

    def save_config(self, config: dict[str, Any]) -> dict[str, Any]:
        merged = {**DEFAULT_CONFIG, **config}
        merged["model_path"] = normalize_model_path(str(merged.get("model_path") or ""), list_candidate_models())
        payload = json.dumps(merged, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO app_state (key, value_json)
                VALUES ('config', ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (payload,),
            )
            self._conn.commit()
        return self.get_config()

    def list_chats(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chat_id, title, created_at, updated_at
                FROM chats
                ORDER BY updated_at DESC, chat_id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_chat(self, title: str | None = None) -> dict[str, Any]:
        safe_title = (title or "New chat").strip() or "New chat"
        with self._lock:
            cur = self._conn.execute("INSERT INTO chats (title) VALUES (?)", (safe_title,))
            self._conn.commit()
            chat_id = cur.lastrowid
        if chat_id is None:
            raise RuntimeError("failed to create chat")
        return self.get_chat(chat_id)

    def ensure_chat(self, chat_id: int | None) -> dict[str, Any]:
        if chat_id is None:
            return self.create_chat()
        return self.get_chat(chat_id)

    def rename_chat_if_placeholder(self, chat_id: int, first_user_message: str) -> None:
        with self._lock:
            row = self._conn.execute("SELECT title FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
        if not row:
            return
        title = str(row["title"] or "").strip()
        if title and title != "New chat":
            return
        new_title = first_user_message.strip().replace("\n", " ")[:80] or "New chat"
        with self._lock:
            self._conn.execute(
                "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?",
                (new_title, chat_id),
            )
            self._conn.commit()

    def add_message(self, chat_id: int, role: str, content: str) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content),
            )
            self._conn.execute("UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?", (chat_id,))
            self._conn.commit()
        return {
            "message_id": cur.lastrowid,
            "chat_id": chat_id,
            "role": role,
            "content": content,
        }

    def get_chat(self, chat_id: int) -> dict[str, Any]:
        with self._lock:
            chat = self._conn.execute(
                "SELECT chat_id, title, created_at, updated_at FROM chats WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        if not chat:
            raise KeyError(f"Chat {chat_id} not found")
        with self._lock:
            messages = self._conn.execute(
                """
                SELECT message_id, role, content, created_at
                FROM messages
                WHERE chat_id = ?
                ORDER BY message_id ASC
                """,
                (chat_id,),
            ).fetchall()
        return {
            **dict(chat),
            "messages": [dict(row) for row in messages],
        }

    def delete_chat(self, chat_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            self._conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            self._conn.commit()
