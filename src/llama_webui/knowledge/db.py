"""Knowledge base SQLite schema and state management."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER,
    ingestion_status TEXT DEFAULT 'pending',
    record_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_ingested_at TIMESTAMP,
    error_message TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_ksource_status ON knowledge_sources(ingestion_status);
CREATE INDEX IF NOT EXISTS idx_ksource_type ON knowledge_sources(source_type);

CREATE TABLE IF NOT EXISTS knowledge_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    external_id TEXT,
    title TEXT,
    record_type TEXT NOT NULL DEFAULT 'conversation',
    category TEXT,
    created_at TIMESTAMP,
    importance_score REAL DEFAULT 0.5,
    metadata TEXT,
    FOREIGN KEY (source_id) REFERENCES knowledge_sources(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_krecord_source ON knowledge_records(source_id);
CREATE INDEX IF NOT EXISTS idx_krecord_category ON knowledge_records(category);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    strategy TEXT,
    metadata TEXT,
    FOREIGN KEY (record_id) REFERENCES knowledge_records(id) ON DELETE CASCADE,
    UNIQUE(record_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_kchunk_record ON knowledge_chunks(record_id);

-- Embedding vectors stored as BLOB (float32 array)
CREATE TABLE IF NOT EXISTS knowledge_embeddings (
    chunk_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL DEFAULT 'llama-server',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES knowledge_chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kemb_model ON knowledge_embeddings(model);

-- FTS5 virtual table for BM25 search
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
    chunk_text,
    content='knowledge_chunks',
    content_rowid='id',
    tokenize='unicode61'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS kchunk_ai AFTER INSERT ON knowledge_chunks BEGIN
    INSERT INTO knowledge_chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS kchunk_ad AFTER DELETE ON knowledge_chunks BEGIN
    INSERT INTO knowledge_chunks_fts(knowledge_chunks_fts, rowid, chunk_text)
        VALUES('delete', old.id, old.chunk_text);
END;

-- Quality metrics
CREATE TABLE IF NOT EXISTS knowledge_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    strategy TEXT,
    bm25_candidates INTEGER DEFAULT 0,
    vector_candidates INTEGER DEFAULT 0,
    final_count INTEGER DEFAULT 0,
    elapsed_ms REAL,
    top_score REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class KnowledgeDB:
    """Thread-safe SQLite knowledge base."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                sources = conn.execute("SELECT COUNT(*) FROM knowledge_sources").fetchone()[0]
                records = conn.execute("SELECT COUNT(*) FROM knowledge_records").fetchone()[0]
                chunks = conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]
                embedded = conn.execute("SELECT COUNT(*) FROM knowledge_embeddings").fetchone()[0]
                fts_ok = True
                try:
                    conn.execute("SELECT COUNT(*) FROM knowledge_chunks_fts LIMIT 1").fetchone()
                except Exception:
                    fts_ok = False
                return {
                    "sources": sources,
                    "records": records,
                    "chunks": chunks,
                    "embedded": embedded,
                    "fts_ok": fts_ok,
                }
            finally:
                conn.close()

    def insert_source(
        self,
        path: str,
        source_type: str,
        file_hash: str,
        file_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO knowledge_sources
                       (path, source_type, file_hash, file_size_bytes, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (path, source_type, file_hash, file_size, json.dumps(metadata or {})),
                )
                if cur.lastrowid and cur.lastrowid > 0:
                    conn.commit()
                    return cur.lastrowid
                row = conn.execute(
                    "SELECT id FROM knowledge_sources WHERE path = ?", (path,)
                ).fetchone()
                return row[0]
            finally:
                conn.close()

    def source_exists_with_hash(self, source_id: int, file_hash: str) -> bool:
        """Check if a source already has this exact hash (unchanged file)."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT file_hash, chunk_count FROM knowledge_sources WHERE id = ?",
                    (source_id,),
                ).fetchone()
                return bool(row and row[0] == file_hash and (row[1] or 0) > 0)
            finally:
                conn.close()

    def update_source_counts(self, source_id: int, record_count: int, chunk_count: int) -> None:
        """Update ingestion counts and timestamp on a source."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """UPDATE knowledge_sources
                       SET record_count = ?, chunk_count = ?,
                           last_ingested_at = CURRENT_TIMESTAMP,
                           ingestion_status = 'completed'
                       WHERE id = ?""",
                    (record_count, chunk_count, source_id),
                )
                conn.commit()
            finally:
                conn.close()

    def insert_record(
        self,
        source_id: int,
        external_id: str | None,
        title: str | None,
        record_type: str,
        category: str | None,
        importance: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO knowledge_records
                       (source_id, external_id, title, record_type, category,
                        importance_score, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_id,
                        external_id,
                        title,
                        record_type,
                        category,
                        importance,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def insert_chunk(
        self,
        record_id: int,
        chunk_index: int,
        text: str,
        chunk_hash: str,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO knowledge_chunks
                       (record_id, chunk_index, chunk_text, chunk_hash, strategy, metadata)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        record_id,
                        chunk_index,
                        text,
                        chunk_hash,
                        strategy,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def insert_embedding(
        self, chunk_id: int, embedding: list[float], model: str = "llama-server"
    ) -> None:
        import struct

        blob = struct.pack(f"{len(embedding)}f", *embedding)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge_embeddings
                       (chunk_id, embedding, model) VALUES (?, ?, ?)""",
                    (chunk_id, blob, model),
                )
                conn.commit()
            finally:
                conn.close()

    def bm25_search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT kc.id, kc.chunk_text, kc.record_id,
                              kr.title, kr.category, ks.source_type
                       FROM knowledge_chunks_fts fts
                       JOIN knowledge_chunks kc ON kc.id = fts.rowid
                       JOIN knowledge_records kr ON kr.id = kc.record_id
                       JOIN knowledge_sources ks ON ks.id = kr.source_id
                       WHERE knowledge_chunks_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (query, limit),
                ).fetchall()
                return [
                    {
                        "chunk_id": r[0],
                        "text": r[1],
                        "record_id": r[2],
                        "title": r[3],
                        "category": r[4],
                        "source": r[5],
                    }
                    for r in rows
                ]
            except Exception:
                return []
            finally:
                conn.close()

    def vector_search(self, query_embedding: list[float], limit: int = 50) -> list[dict[str, Any]]:
        import struct

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT chunk_id, embedding FROM knowledge_embeddings"
                ).fetchall()
            finally:
                conn.close()

        candidates: list[tuple[float, int]] = []
        for chunk_id, emb_blob in rows:
            stored = list(struct.unpack(f"{len(emb_blob) // 4}f", emb_blob))
            score = _cosine_similarity(query_embedding, stored)
            candidates.append((score, chunk_id))

        candidates.sort(reverse=True)
        top_ids = [c[1] for c in candidates[:limit]]
        score_map = {c[1]: c[0] for c in candidates[:limit]}

        if not top_ids:
            return []

        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(top_ids))
                rows = conn.execute(
                    f"""SELECT kc.id, kc.chunk_text, kc.record_id,
                              kr.title, kr.category, ks.source_type
                       FROM knowledge_chunks kc
                       JOIN knowledge_records kr ON kr.id = kc.record_id
                       JOIN knowledge_sources ks ON ks.id = kr.source_id
                       WHERE kc.id IN ({placeholders})""",
                    top_ids,
                ).fetchall()
            finally:
                conn.close()

        return [
            {
                "chunk_id": r[0],
                "text": r[1],
                "record_id": r[2],
                "title": r[3],
                "category": r[4],
                "source": r[5],
                "vector_score": score_map.get(r[0], 0.0),
            }
            for r in rows
        ]

    def get_chunks_for_embedding(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT kc.id, kc.chunk_text
                       FROM knowledge_chunks kc
                       WHERE NOT EXISTS (
                           SELECT 1 FROM knowledge_embeddings ke WHERE ke.chunk_id = kc.id
                       )
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
                return [{"chunk_id": r[0], "text": r[1]} for r in rows]
            finally:
                conn.close()

    def record_metric(
        self,
        query: str,
        strategy: str,
        bm25: int,
        vector: int,
        final: int,
        elapsed_ms: float,
        top_score: float,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO knowledge_metrics
                       (query, strategy, bm25_candidates, vector_candidates,
                        final_count, elapsed_ms, top_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (query, strategy, bm25, vector, final, elapsed_ms, top_score),
                )
                conn.commit()
            finally:
                conn.close()

    def clear_all(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM knowledge_embeddings")
                conn.execute("DELETE FROM knowledge_chunks")
                conn.execute("DELETE FROM knowledge_records")
                conn.execute("DELETE FROM knowledge_sources")
                conn.execute("DELETE FROM knowledge_metrics")
                conn.commit()
            finally:
                conn.close()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
