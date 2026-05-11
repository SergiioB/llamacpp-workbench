"""Embedding generation via llama-server /v1/embeddings endpoint."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using the managed llama-server instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8085) -> None:
        self.base_url = f"http://{host}:{port}"

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float] | None:
        payload = json.dumps(
            {
                "input": text,
                "model": "local",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                data = body.get("data", [])
                if data and isinstance(data[0], dict):
                    return data[0].get("embedding")
        except urllib.error.HTTPError as e:
            logger.warning(
                "Embedding request failed (HTTP %d): %s", e.code, e.read().decode()[:200]
            )
        except Exception as e:
            logger.warning("Embedding request failed: %s", e)
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        results: list[list[float] | None] = []
        for text in texts:
            results.append(self.embed(text))
        return results


def embed_chunks_for_db(
    db: Any,
    embedder: Embedder,
    batch_size: int = 32,
) -> dict[str, int]:
    """Embed all pending chunks and store in DB. Returns stats."""
    embedded = 0
    failed = 0
    while True:
        pending = db.get_chunks_for_embedding(limit=batch_size)
        if not pending:
            break
        for chunk in pending:
            vec = embedder.embed(chunk["text"])
            if vec:
                db.insert_embedding(chunk["chunk_id"], vec)
                embedded += 1
            else:
                failed += 1
    return {"embedded": embedded, "failed": failed}
