"""Hybrid retrieval pipeline — BM25 (FTS5) + vector similarity + reranking."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .db import KnowledgeDB
from .embedder import Embedder


@dataclass
class RetrievalResult:
    chunk_id: int
    text: str
    title: str
    source: str
    category: str
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:500],
            "title": self.title,
            "source": self.source,
            "category": self.category,
            "score": round(self.score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "vector_score": round(self.vector_score, 4),
            "matched_terms": self.matched_terms,
        }


STOP_WORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "what",
    "which",
    "who",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "to",
    "from",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "own",
    "so",
    "than",
    "too",
    "very",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[a-z0-9]{3,}", normalized)
    return [t for t in tokens if t not in STOP_WORDS]


def build_fts_query(query: str) -> str:
    tokens = list(dict.fromkeys(tokenize(query)))  # deduplicate preserving order
    if not tokens:
        return ""
    return " OR ".join(tokens)


def retrieve(
    query: str,
    db: KnowledgeDB,
    embedder: Embedder,
    top_k: int = 10,
    use_vectors: bool = True,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    """Run hybrid retrieval: BM25 + optional vector search, merge and rank."""
    t0 = time.perf_counter()
    stats: dict[str, Any] = {
        "query": query,
        "bm25_candidates": 0,
        "vector_candidates": 0,
        "final_count": 0,
        "elapsed_ms": 0,
    }

    tokens = tokenize(query)
    results_map: dict[int, RetrievalResult] = {}

    # Stage 1: BM25 via FTS5
    fts_query = build_fts_query(query)
    bm25_rows = db.bm25_search(fts_query, limit=top_k * 5) if fts_query else []
    stats["bm25_candidates"] = len(bm25_rows)
    for row in bm25_rows:
        text_lower = row["text"].lower()
        matched = [t for t in tokens if t in text_lower]
        if not matched:
            continue
        match_ratio = len(matched) / max(len(tokens), 1)
        tf_score = sum(min(text_lower.count(t) * 0.5, 2.0) for t in matched)
        bm25_score = match_ratio * 3.0 + tf_score
        results_map[row["chunk_id"]] = RetrievalResult(
            chunk_id=row["chunk_id"],
            text=row["text"],
            title=row["title"] or "Untitled",
            source=row["source"],
            category=row["category"] or "unknown",
            bm25_score=bm25_score,
            score=bm25_score * 0.6,
            matched_terms=sorted(set(matched)),
        )

    # Stage 2: Vector search
    if use_vectors:
        query_vec = embedder.embed(query)
        if query_vec:
            vec_rows = db.vector_search(query_vec, limit=top_k * 5)
            stats["vector_candidates"] = len(vec_rows)
            for row in vec_rows:
                vec_score = row.get("vector_score", 0.0)
                cid = row["chunk_id"]
                if cid in results_map:
                    existing = results_map[cid]
                    existing.vector_score = vec_score
                    existing.score = (existing.bm25_score * 0.6 + vec_score * 0.4) * 1.3
                else:
                    results_map[cid] = RetrievalResult(
                        chunk_id=cid,
                        text=row["text"],
                        title=row["title"] or "Untitled",
                        source=row["source"],
                        category=row["category"] or "unknown",
                        vector_score=vec_score,
                        score=vec_score * 0.4,
                    )

    # Sort by combined score
    results = sorted(results_map.values(), key=lambda r: r.score, reverse=True)[:top_k]
    stats["final_count"] = len(results)
    stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Record metric
    top_score = results[0].score if results else 0.0
    db.record_metric(
        query=query,
        strategy="hybrid" if use_vectors else "bm25",
        bm25=stats["bm25_candidates"],
        vector=stats["vector_candidates"],
        final=stats["final_count"],
        elapsed_ms=stats["elapsed_ms"],
        top_score=top_score,
    )

    return results, stats
