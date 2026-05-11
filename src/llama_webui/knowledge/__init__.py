"""Knowledge base module — RAG pipeline for llama-webui."""

from .chunker import Chunk, chunk_text
from .db import KnowledgeDB
from .embedder import Embedder
from .ingest import IngestResult, ingest_directory, ingest_jsonl
from .retriever import RetrievalResult, retrieve

__all__ = [
    "KnowledgeDB",
    "chunk_text",
    "Chunk",
    "Embedder",
    "retrieve",
    "RetrievalResult",
    "ingest_jsonl",
    "ingest_directory",
    "IngestResult",
]
