"""Smart chunking strategies — conversation-aware, recursive, semantic, fixed, sliding."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkStrategy(Enum):
    FIXED = "fixed"
    SLIDING = "sliding"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    CONVERSATION = "conversation"


@dataclass
class Chunk:
    text: str
    index: int
    start: int = 0
    end: int = 0
    strategy: str = ""
    hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.text.encode()).hexdigest()
        if self.end == 0 and self.text:
            self.end = len(self.text)


def chunk_fixed(text: str, size: int = 512, min_size: int = 50) -> list[Chunk]:
    if not text or len(text.strip()) < min_size:
        return []
    if len(text) <= size:
        return [Chunk(text=text.strip(), index=0, strategy=ChunkStrategy.FIXED.value)]
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if len(piece) >= min_size:
            chunks.append(
                Chunk(
                    text=piece, index=idx, start=start, end=end, strategy=ChunkStrategy.FIXED.value
                )
            )
            idx += 1
        start = end
    return chunks


def chunk_sliding(text: str, size: int = 512, overlap: int = 64, min_size: int = 50) -> list[Chunk]:
    if not text or len(text.strip()) < min_size:
        return []
    if len(text) <= size:
        return [Chunk(text=text.strip(), index=0, strategy=ChunkStrategy.SLIDING.value)]
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            search_start = start + int(size * 0.8)
            for bp in ["\n\n", ". ", "。", "\n", "! ", "? "]:
                pos = text.rfind(bp, search_start, end)
                if pos > search_start:
                    end = pos + len(bp)
                    break
        piece = text[start:end].strip()
        if len(piece) >= min_size:
            chunks.append(
                Chunk(
                    text=piece,
                    index=idx,
                    start=start,
                    end=end,
                    strategy=ChunkStrategy.SLIDING.value,
                )
            )
            idx += 1
        start = max(end - overlap, start + 1) if end < len(text) else end
    return chunks


def chunk_recursive(text: str, max_size: int = 512, min_size: int = 50) -> list[Chunk]:
    if not text or len(text.strip()) < min_size:
        return []
    if len(text) <= max_size:
        return [Chunk(text=text.strip(), index=0, strategy=ChunkStrategy.RECURSIVE.value)]
    chunks: list[Chunk] = []
    idx = [0]

    def _split(block: str, offset: int) -> None:
        if len(block) <= max_size:
            if len(block.strip()) >= min_size:
                chunks.append(
                    Chunk(
                        text=block.strip(),
                        index=idx[0],
                        start=offset,
                        end=offset + len(block),
                        strategy=ChunkStrategy.RECURSIVE.value,
                    )
                )
                idx[0] += 1
            return
        for sep in ["\n\n", "\n", ". ", "。", "! ", "? ", "; ", " "]:
            parts = block.split(sep)
            if len(parts) < 2:
                continue
            current = ""
            cur_start = offset
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current and len(current.strip()) >= min_size:
                        chunks.append(
                            Chunk(
                                text=current.strip(),
                                index=idx[0],
                                start=cur_start,
                                end=cur_start + len(current),
                                strategy=ChunkStrategy.RECURSIVE.value,
                            )
                        )
                        idx[0] += 1
                    current = part
                    cur_start = offset + block.find(part, len(current))
            if current and len(current.strip()) >= min_size:
                chunks.append(
                    Chunk(
                        text=current.strip(),
                        index=idx[0],
                        start=cur_start,
                        end=cur_start + len(current),
                        strategy=ChunkStrategy.RECURSIVE.value,
                    )
                )
                idx[0] += 1
            return
        for i in range(0, len(block), max_size):
            piece = block[i : i + max_size].strip()
            if len(piece) >= min_size:
                chunks.append(
                    Chunk(
                        text=piece,
                        index=idx[0],
                        start=offset + i,
                        end=offset + i + len(piece),
                        strategy=ChunkStrategy.RECURSIVE.value,
                    )
                )
                idx[0] += 1

    _split(text, 0)
    for i, c in enumerate(chunks):
        c.index = i
    return chunks


def chunk_semantic(text: str, max_size: int = 512, min_size: int = 50) -> list[Chunk]:
    if not text or len(text.strip()) < min_size:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return []
    chunks: list[Chunk] = []
    current = ""
    cur_start = 0
    idx = 0
    for para in paragraphs:
        para_start = text.find(para, cur_start)
        _is_heading = bool(re.match(r"^#{1,6}\s", para)) or (len(para) < 80 and para.endswith(":"))
        candidate = current + "\n\n" + para if current else para
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current and len(current.strip()) >= min_size:
                chunks.append(
                    Chunk(
                        text=current.strip(),
                        index=idx,
                        start=cur_start,
                        end=cur_start + len(current),
                        strategy=ChunkStrategy.SEMANTIC.value,
                    )
                )
                idx += 1
            current = para
            cur_start = para_start
    if current and len(current.strip()) >= min_size:
        chunks.append(
            Chunk(
                text=current.strip(),
                index=idx,
                start=cur_start,
                end=cur_start + len(current),
                strategy=ChunkStrategy.SEMANTIC.value,
            )
        )
    return chunks


def chunk_conversation(messages: list[dict[str, str]], max_chars: int = 2000) -> list[Chunk]:
    if not messages:
        return []
    chunks: list[Chunk] = []
    current_turns: list[dict[str, str]] = []
    current_text = ""
    idx = 0
    char_offset = 0
    cur_start = 0
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not content:
            continue
        turn = f"{role}: {content}\n"
        candidate = current_text + turn
        if len(candidate) > max_chars and current_turns:
            chunks.append(
                Chunk(
                    text=current_text.strip(),
                    index=idx,
                    start=cur_start,
                    end=char_offset,
                    strategy=ChunkStrategy.CONVERSATION.value,
                )
            )
            idx += 1
            current_turns = []
            current_text = turn
            cur_start = char_offset
        else:
            current_text = candidate
        current_turns.append(msg)
        char_offset += len(turn)
    if current_text and current_text.strip():
        chunks.append(
            Chunk(
                text=current_text.strip(),
                index=idx,
                start=cur_start,
                end=char_offset,
                strategy=ChunkStrategy.CONVERSATION.value,
            )
        )
    return chunks


CONVERSATION_SOURCES = {
    "pi",
    "claude",
    "codex",
    "gemini",
    "droid",
    "factory",
    "qwen_code",
    "opencode",
    "cursor",
    "hermes",
    "openclaw",
}


def chunk_text(
    text: str,
    source_type: str = "unknown",
    strategy: str = "auto",
    chunk_size: int = 512,
    overlap: int = 64,
    min_size: int = 50,
) -> list[Chunk]:
    if not text or len(text.strip()) < min_size:
        return []

    if strategy == "auto":
        if source_type in CONVERSATION_SOURCES:
            msgs: list[dict[str, str]] = []
            for m in re.finditer(
                r"(user|assistant|human|ai|tool|system):\s(.*?)(?=\n(?:user|assistant|human|ai|tool|system):|$)",
                text,
                re.DOTALL | re.IGNORECASE,
            ):
                msgs.append({"role": m.group(1).lower(), "content": m.group(2).strip()})
            if len(msgs) >= 2:
                return chunk_conversation(msgs, max_chars=chunk_size)
        heading_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        if heading_count >= 3:
            return chunk_semantic(text, chunk_size, min_size)
        code_blocks = len(re.findall(r"```", text))
        if code_blocks >= 4 or source_type == "code":
            return chunk_fixed(text, chunk_size, min_size)
        return chunk_recursive(text, chunk_size, min_size)

    if strategy == "fixed":
        return chunk_fixed(text, chunk_size, min_size)
    if strategy == "sliding":
        return chunk_sliding(text, chunk_size, overlap, min_size)
    if strategy == "recursive":
        return chunk_recursive(text, chunk_size, min_size)
    if strategy == "semantic":
        return chunk_semantic(text, chunk_size, min_size)
    if strategy == "conversation":
        msgs = []
        for m in re.finditer(
            r"(user|assistant|human|ai|tool|system):\s(.*?)(?=\n(?:user|assistant|human|ai|tool|system):|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        ):
            msgs.append({"role": m.group(1).lower(), "content": m.group(2).strip()})
        return (
            chunk_conversation(msgs, max_chars=chunk_size)
            if len(msgs) >= 2
            else chunk_sliding(text, chunk_size, overlap, min_size)
        )
    return chunk_sliding(text, chunk_size, overlap, min_size)
