"""Knowledge base module — RAG pipeline for llama-webui."""

from .chunker import Chunk, chunk_text
from .db import KnowledgeDB
from .embedder import Embedder
from .ingest import (
    IngestResult,
    discover_sources,
    ingest_directory,
    ingest_jsonl,
    ingest_opencode_db,
)
from .retriever import RetrievalResult, retrieve

__all__ = [
    "KnowledgeDB",
    "chunk_text",
    "Chunk",
    "Embedder",
    "retrieve",
    "RetrievalResult",
    "discover_sources",
    "ingest_jsonl",
    "ingest_directory",
    "ingest_opencode_db",
    "IngestResult",
]
