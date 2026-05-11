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

# Source-specific discovery paths
SOURCE_PATHS: dict[str, list[Path]] = {
    "pi": [Path.home() / ".pi" / "agent" / "sessions"],
    "claude": [Path.home() / ".claude" / "projects"],
    "codex": [Path.home() / ".codex" / "sessions"],
    "factory": [Path.home() / ".factory" / "sessions"],
    "opencode": [Path.home() / ".local" / "share" / "opencode"],
    "qwen_code": [Path.home() / ".qwen" / "projects"],
}


@dataclass
class IngestResult:
    source: str
    path: str
    records: int = 0
    chunks: int = 0
    embedded: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def _parse_jsonl_messages(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract messages from various JSONL conversation formats."""
    messages: list[dict[str, str]] = []
    raw_messages = data.get("messages") or data.get("turns") or []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("sender") or "unknown"
        content = msg.get("content") or msg.get("text") or msg.get("message") or ""
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        content = str(content).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _extract_title(data: dict[str, Any], path: Path) -> str:
    return (
        data.get("title")
        or data.get("name")
        or data.get("session_title")
        or data.get("session_display_title")
        or path.stem
    )


def ingest_jsonl(
    path: Path,
    source_type: str,
    db: KnowledgeDB,
    embedder: Embedder,
    chunk_size: int = 512,
    embed: bool = True,
) -> IngestResult:
    """Ingest a single JSONL file containing conversations."""
    result = IngestResult(source=source_type, path=str(path))
    if not path.exists():
        result.errors.append("File not found")
        return result

    file_hash = _file_hash(path)
    file_size = path.stat().st_size
    source_id = db.insert_source(
        str(path),
        source_type,
        file_hash,
        file_size,
        metadata={"type": "jsonl"},
    )

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = _parse_jsonl_messages(data)
            if not messages:
                continue

            title = _extract_title(data, path)
            conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            category = "chat"
            lower_text = conversation_text.lower()
            if any(kw in lower_text for kw in ["def ", "class ", "import ", "```"]):
                category = "code"
            elif any(kw in lower_text for kw in ["tool", "execute", "script"]):
                category = "tool"

            importance = 0.5
            if category == "code":
                importance = 0.7
            elif category == "tool":
                importance = 0.6
            if len(conversation_text) > 1000:
                importance += 0.1

            external_id = (
                data.get("id")
                or data.get("conversation_id")
                or data.get("session_id")
                or f"{path.stem}:{line_num}"
            )

            record_id = db.insert_record(
                source_id=source_id,
                external_id=str(external_id),
                title=title,
                record_type="conversation",
                category=category,
                importance=min(importance, 1.0),
                metadata={
                    "message_count": len(messages),
                    "line_num": line_num,
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
    discovered: list[dict[str, Any]] = []
    for source_name, paths in SOURCE_PATHS.items():
        for base_path in paths:
            if base_path.exists():
                jsonl_files = list(base_path.rglob("*.jsonl"))
                json_files = list(base_path.rglob("*.json"))
                total = len(jsonl_files) + len(json_files)
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
