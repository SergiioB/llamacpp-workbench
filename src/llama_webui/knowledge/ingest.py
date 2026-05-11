"""Ingestion pipeline — discover, extract, chunk, and embed sources."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import chunk_text
from .db import KnowledgeDB
from .embedder import Embedder, embed_chunks_for_db

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: str
    path: str
    records: int = 0
    chunks: int = 0
    embedded: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from various content block formats.

    Handles:
    - str: returned as-is
    - list[dict]: content block arrays (Pi, Codex, Factory, Qwen)
      - {"type": "text", "text": "..."}
      - {"type": "input_text", "text": "..."}
      - {"type": "thinking", "thinking": "..."}  (skipped)
      - {"text": "...", "thought": true}  (Qwen — skipped when thinking)
      - {"text": "...", "thought": false}  (Qwen — kept)
      - {"text": "..."}  (Qwen user — no type, no thought)
      - {"functionCall": {...}}  (skipped)
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip() if content else ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue

        block_type = block.get("type", "")

        # Qwen-style: {"text": "...", "thought": true/false}
        # Skip thinking blocks
        if block.get("thought"):
            continue

        # Standard text blocks
        if (
            block_type in ("text", "input_text")
            or not block_type
            and "text" in block
            and "thought" not in block
            or "text" in block
            and not block.get("thought")
            and not block.get("functionCall")
        ):
            text = block.get("text", "")
            if text:
                parts.append(text)

        # Skip function_call / tool_use blocks

    return "\n".join(parts).strip()


def _parse_pi_session(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Parse Pi agent session JSONL (event-stream format).

    Format: each line has "type" field.
    Messages have type="message" with nested message.content as array of blocks.
    """
    messages: list[dict[str, str]] = []
    for line in lines:
        if line.get("type") != "message":
            continue
        msg = line.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = _extract_text_from_content(msg.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _parse_codex_session(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Parse Codex session JSONL (event-stream format).

    Format: lines with type="response_item", payload.type="message",
    payload.content as array of content blocks.
    """
    messages: list[dict[str, str]] = []
    for line in lines:
        if line.get("type") != "response_item":
            continue
        payload = line.get("payload", {})
        if not isinstance(payload, dict) or payload.get("type") != "message":
            continue
        role = payload.get("role", "unknown")
        content = _extract_text_from_content(payload.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _parse_factory_session(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Parse Factory/DROID session JSONL (event-stream format).

    Same structure as Pi: type="message" with message.content as array.
    """
    return _parse_pi_session(lines)  # Same format


def _parse_qwen_code_session(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Parse Qwen Code session JSONL.

    Format: lines with type="user" or type="assistant",
    message.parts is array of {"text": "...", "thought": true/false} blocks.
    Uses role="model" instead of "assistant".
    """
    messages: list[dict[str, str]] = []
    for line in lines:
        line_type = line.get("type", "")
        if line_type not in ("user", "assistant"):
            continue
        msg = line.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", line_type)
        # Normalize "model" -> "assistant"
        if role == "model":
            role = "assistant"
        # Qwen uses "parts" instead of "content"
        parts = msg.get("parts") or msg.get("content") or []
        content = _extract_text_from_content(parts)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _parse_generic_jsonl(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Fallback: try to extract messages from any JSONL format.

    Tries common field names: messages, turns, or per-line message detection.
    """
    messages: list[dict[str, str]] = []

    # If any line has a top-level "messages" or "turns" array, use that
    for line in lines:
        raw_msgs = line.get("messages") or line.get("turns") or []
        if isinstance(raw_msgs, list) and raw_msgs:
            for msg in raw_msgs:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role") or msg.get("sender") or "unknown"
                content = _extract_text_from_content(
                    msg.get("content") or msg.get("text") or msg.get("message") or ""
                )
                if content:
                    messages.append({"role": role, "content": content})
            return messages  # Found a batch-format line, return what we got

    # Per-line message detection
    for line in lines:
        role = line.get("role", "")
        if role in ("user", "assistant", "human", "ai", "system", "developer"):
            content = _extract_text_from_content(
                line.get("content") or line.get("text") or line.get("message") or ""
            )
            if content:
                messages.append({"role": role, "content": content})

    return messages


def _parse_session(lines: list[dict[str, Any]], source_type: str) -> list[dict[str, str]]:
    """Route to the correct parser based on source type."""
    parsers = {
        "pi": _parse_pi_session,
        "claude": _parse_pi_session,  # Claude uses similar event-stream
        "codex": _parse_codex_session,
        "factory": _parse_factory_session,
        "droid": _parse_factory_session,
        "qwen_code": _parse_qwen_code_session,
    }
    parser = parsers.get(source_type)
    if parser:
        return parser(lines)
    return _parse_generic_jsonl(lines)


def _extract_session_title(lines: list[dict[str, Any]], path: Path) -> str:
    """Try to extract a session title from the first few event lines."""
    for line in lines[:20]:
        title = line.get("title") or line.get("sessionTitle") or line.get("session_display_title")
        if title and isinstance(title, str) and len(title) > 2:
            return title
        # Pi session events may have title in the first line
        if line.get("type") == "session":
            session_title = line.get("title") or ""
            if session_title:
                return session_title
        # Factory session_start
        if line.get("type") == "session_start":
            session_title = line.get("title") or line.get("sessionTitle") or ""
            if session_title:
                return session_title
    return path.stem


def _extract_session_id(lines: list[dict[str, Any]]) -> str:
    """Extract a session ID from event-stream lines."""
    for line in lines[:5]:
        sid = line.get("id") or line.get("sessionId") or ""
        if sid and isinstance(sid, str):
            return sid
    return ""


def ingest_jsonl(
    path: Path,
    source_type: str,
    db: KnowledgeDB,
    embedder: Embedder,
    chunk_size: int = 512,
    embed: bool = True,
) -> IngestResult:
    """Ingest a single JSONL file as one session/conversation.

    Handles event-stream JSONL formats (Pi, Codex, Factory, Qwen Code)
    where each line is an event and messages span multiple lines.
    """
    result = IngestResult(source=source_type, path=str(path))
    if not path.exists():
        result.errors.append("File not found")
        return result

    file_hash = _file_hash(path)
    file_size = path.stat().st_size

    # Skip very small files (< 100 bytes, likely empty or metadata-only)
    if file_size < 100:
        return result

    # Skip if this exact file was already ingested (same path + hash)
    source_id = db.insert_source(
        str(path),
        source_type,
        file_hash,
        file_size,
        metadata={"type": "jsonl", "format": "event_stream"},
    )
    # insert_source returns existing ID if path matches, but hash may differ.
    # If hash matches, the file content hasn't changed — skip re-ingest.
    if db.source_exists_with_hash(source_id, file_hash):
        return result

    # Read and parse all lines
    lines: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not lines:
        return result

    # Extract messages using source-specific parser
    messages = _parse_session(lines, source_type)
    if not messages:
        return result

    # Filter out system/developer messages — they're instructions, not conversations
    conversation_messages = [
        m for m in messages if m["role"] in ("user", "assistant", "human", "ai")
    ]
    if not conversation_messages:
        return result

    title = _extract_session_title(lines, path)
    conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation_messages)

    # Skip if too short after filtering
    if len(conversation_text.strip()) < 50:
        return result

    # Classify content
    category = "chat"
    lower_text = conversation_text.lower()
    if any(kw in lower_text for kw in ["def ", "class ", "import ", "```"]):
        category = "code"
    elif any(kw in lower_text for kw in ["tool", "execute", "script", "terminal"]):
        category = "tool"

    importance = 0.5
    if category == "code":
        importance = 0.7
    elif category == "tool":
        importance = 0.6
    if len(conversation_text) > 1000:
        importance += 0.1
    if len(conversation_messages) > 10:
        importance += 0.1

    external_id = _extract_session_id(lines) or path.stem

    record_id = db.insert_record(
        source_id=source_id,
        external_id=external_id,
        title=title,
        record_type="conversation",
        category=category,
        importance=min(importance, 1.0),
        metadata={
            "message_count": len(conversation_messages),
            "file_size": file_size,
        },
    )
    result.records += 1

    chunks = chunk_text(
        conversation_text,
        source_type=source_type,
        strategy="auto",
        chunk_size=chunk_size,
    )
    for chunk in chunks:
        db.insert_chunk(
            record_id=record_id,
            chunk_index=chunk.index,
            text=chunk.text,
            chunk_hash=chunk.hash,
            strategy=chunk.strategy,
            metadata=chunk.metadata,
        )
        result.chunks += 1

    # Update source with actual chunk count
    db.update_source_counts(source_id, result.records, result.chunks)

    if embed and result.chunks > 0 and embedder.is_available():
        embed_stats = embed_chunks_for_db(db, embedder, batch_size=32)
        result.embedded = embed_stats["embedded"]

    return result


def ingest_directory(
    directory: Path,
    source_type: str,
    db: KnowledgeDB,
    embedder: Embedder,
    pattern: str = "*.jsonl",
    chunk_size: int = 512,
    embed: bool = True,
) -> list[IngestResult]:
    """Ingest all matching files from a directory."""
    results: list[IngestResult] = []
    if not directory.exists():
        return [
            IngestResult(source=source_type, path=str(directory), errors=["Directory not found"])
        ]

    files = sorted(directory.rglob(pattern))
    if not files:
        return [
            IngestResult(
                source=source_type, path=str(directory), errors=["No files matched pattern"]
            )
        ]

    for path in files:
        logger.info("Ingesting %s (%s)", path.name, source_type)
        result = ingest_jsonl(path, source_type, db, embedder, chunk_size, embed)
        results.append(result)

    return results


def discover_sources(_db: KnowledgeDB) -> list[dict[str, Any]]:
    """Auto-discover known AI tool session directories."""
    from ..settings import knowledge_source_paths

    source_paths = knowledge_source_paths()
    discovered: list[dict[str, Any]] = []
    for source_name, paths in source_paths.items():
        for base_path in paths:
            if base_path.exists():
                jsonl_files = list(base_path.rglob("*.jsonl"))
                json_files = list(base_path.rglob("*.json"))
                db_files = list(base_path.rglob("*.db"))
                total = len(jsonl_files) + len(json_files) + len(db_files)
                discovered.append(
                    {
                        "source": source_name,
                        "path": str(base_path),
                        "available": True,
                        "files": total,
                    }
                )
            else:
                discovered.append(
                    {
                        "source": source_name,
                        "path": str(base_path),
                        "available": False,
                        "files": 0,
                    }
                )
    return discovered


def ingest_opencode_db(
    db_path: Path,
    db: KnowledgeDB,
    embedder: Embedder,
    chunk_size: int = 512,
    embed: bool = True,
) -> IngestResult:
    """Ingest conversations from an OpenCode SQLite database.

    OpenCode stores sessions in SQLite with session/message/part tables.
    """
    result = IngestResult(source="opencode", path=str(db_path))
    if not db_path.exists():
        result.errors.append("Database file not found")
        return result

    import sqlite3 as sqlite3_mod

    file_hash = _file_hash(db_path)
    file_size = db_path.stat().st_size

    source_id = db.insert_source(
        str(db_path),
        "opencode",
        file_hash,
        file_size,
        metadata={"type": "sqlite"},
    )
    if db.source_exists_with_hash(source_id, file_hash):
        return result

    try:
        src_conn = sqlite3_mod.connect(str(db_path))
        sessions = src_conn.execute("SELECT id, title FROM session").fetchall()
    except Exception as e:
        result.errors.append(f"Failed to read OpenCode DB: {e}")
        return result
    finally:
        src_conn.close()

    src_conn = sqlite3_mod.connect(str(db_path), check_same_thread=False)
    try:
        for session_id, session_title in sessions:
            # Get messages with their text parts
            rows = src_conn.execute(
                """SELECT m.id, COALESCE(json_extract(m.data, '$.role'), 'unknown'),
                          GROUP_CONCAT(
                            CASE WHEN json_extract(p.data, '$.type') = 'text'
                                 THEN json_extract(p.data, '$.text') END, '\n'
                          )
                   FROM message m
                   LEFT JOIN part p ON p.message_id = m.id
                   WHERE m.session_id = ?
                   GROUP BY m.id
                   ORDER BY m.time_created""",
                (session_id,),
            ).fetchall()

            messages: list[dict[str, str]] = []
            for _msg_id, role, text in rows:
                if not text or not text.strip():
                    continue
                if role not in ("user", "assistant"):
                    continue
                messages.append({"role": role, "content": text.strip()})

            if not messages:
                continue

            conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            if len(conversation_text.strip()) < 50:
                continue

            category = "chat"
            lower = conversation_text.lower()
            if any(kw in lower for kw in ["def ", "class ", "import ", "```"]):
                category = "code"
            elif any(kw in lower for kw in ["tool", "execute", "script"]):
                category = "tool"

            importance = 0.5
            if category == "code":
                importance = 0.7
            elif category == "tool":
                importance = 0.6
            if len(conversation_text) > 1000:
                importance += 0.1

            record_id = db.insert_record(
                source_id=source_id,
                external_id=session_id,
                title=session_title or "OpenCode Session",
                record_type="conversation",
                category=category,
                importance=min(importance, 1.0),
                metadata={"message_count": len(messages)},
            )
            result.records += 1

            chunks = chunk_text(
                conversation_text,
                source_type="opencode",
                strategy="auto",
                chunk_size=chunk_size,
            )
            for chunk in chunks:
                db.insert_chunk(
                    record_id=record_id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    chunk_hash=chunk.hash,
                    strategy=chunk.strategy,
                    metadata=chunk.metadata,
                )
                result.chunks += 1
    finally:
        src_conn.close()

    db.update_source_counts(source_id, result.records, result.chunks)

    if embed and result.chunks > 0 and embedder.is_available():
        embed_stats = embed_chunks_for_db(db, embedder, batch_size=32)
        result.embedded = embed_stats["embedded"]

    return result
