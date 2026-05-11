from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .app_state import AppState
from .download_manager import ModelDownloadManager
from .knowledge.db import KnowledgeDB
from .knowledge.embedder import Embedder, embed_chunks_for_db
from .knowledge.ingest import discover_sources, ingest_directory, ingest_jsonl
from .knowledge.retriever import retrieve
from .llama_manager import LlamaServerManager
from .model_inventory import (
    apply_model_profile,
    build_model_presets,
    list_candidate_models,
    scan_models,
)
from .preflight import run_preflight
from .settings import PROJECT_ROOT, STATIC_DIR, data_dir, default_download_dir

state = AppState(PROJECT_ROOT)
manager = LlamaServerManager(state.log_path)
downloads = ModelDownloadManager(data_dir() / "downloads", default_download_dir())
knowledge_db = KnowledgeDB(data_dir() / "knowledge.db")


def _embedder_from_config(config: dict[str, Any]) -> Embedder:
    return Embedder(
        host=str(config.get("llama_host") or "127.0.0.1"),
        port=int(config.get("llama_port") or 8085),
    )


app = FastAPI(title="llama-webui")


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Add no-cache headers to static assets so UI updates are always visible."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheStaticMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ConfigPayload(BaseModel):
    config: dict[str, Any]


class StartPayload(BaseModel):
    config: dict[str, Any] | None = None


class RpcPreflightPayload(BaseModel):
    config: dict[str, Any] | None = None


class PreflightPayload(BaseModel):
    config: dict[str, Any] | None = None


class ChatRenamePayload(BaseModel):
    title: str


class ChatCreatePayload(BaseModel):
    title: str | None = None


class MessagePayload(BaseModel):
    content: str


class DownloadPayload(BaseModel):
    url: str
    destination_path: str | None = None


class ModelLoadPayload(BaseModel):
    model_path: str


class KnowledgeQueryPayload(BaseModel):
    query: str
    top_k: int = 10
    use_vectors: bool = True


class KnowledgeIngestPayload(BaseModel):
    source: str = "pi"
    path: str | None = None
    pattern: str = "*.jsonl"
    chunk_size: int = 512
    embed: bool = True


class KnowledgeEmbedPayload(BaseModel):
    batch_size: int = 32


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    config = state.get_config()
    candidates = list_candidate_models()
    return {
        "config": config,
        "candidate_models": candidates,
        "model_presets": build_model_presets(config, candidates),
    }


@app.post("/api/config")
def save_config(payload: ConfigPayload) -> dict[str, Any]:
    return {"config": state.save_config(payload.config)}


@app.get("/api/server/status")
def server_status() -> dict[str, Any]:
    config = state.get_config()
    status = manager.health(config)
    return {"config": config, "status": status}


@app.post("/api/rpc/preflight")
def rpc_preflight(payload: RpcPreflightPayload) -> dict[str, Any]:
    config = {**state.get_config(), **(payload.config or {})}
    status = manager.health(config)
    if status.get("managed") and status.get("pid"):
        host = str(config.get("rpc_host") or "").strip()
        try:
            port = int(config.get("rpc_port") or 0)
        except (TypeError, ValueError):
            port = 0
        endpoint = f"{host}:{port}" if host and port > 0 else ""
        return {"enabled": True, "reachable": True, "endpoint": endpoint, "active": True}
    return manager.rpc_preflight(config)


async def _bg_start(config: dict[str, Any]) -> None:
    """Run manager.start in a worker thread so we don't block the event loop."""
    import asyncio

    loop = asyncio.get_event_loop()
    with suppress(Exception):
        await loop.run_in_executor(None, manager.start, config)


@app.post("/api/preflight")
def preflight_check(payload: PreflightPayload | None = None) -> dict[str, Any]:
    """Run all preflight checks and return readiness status."""
    config = {**state.get_config(), **((payload.config if payload else None) or {})}
    status = manager.health(config)
    if status.get("managed") and status.get("pid"):
        if status.get("healthy"):
            return {
                "ready": True,
                "checks": {"server": status},
                "blocking_issues": [],
                "warnings": [],
                "log_diagnoses": [],
            }
        return {
            "ready": False,
            "checks": {"server": status},
            "blocking_issues": [],
            "warnings": ["llama-server is starting; readiness checks are paused."],
            "log_diagnoses": [],
        }
    return run_preflight(config, log_path=manager.log_path)


@app.get("/api/server/logs")
def server_logs(lines: int = 80) -> dict[str, Any]:
    """Return last N lines of llama-server log with parsed diagnoses."""
    from .preflight import parse_log_errors

    log_text = ""
    if manager.log_path.exists():
        try:
            raw = manager.log_path.read_text(encoding="utf-8", errors="replace")
            all_lines = raw.splitlines()
            log_text = "\n".join(all_lines[-lines:])
        except OSError:
            pass

    diagnoses = parse_log_errors(log_text)
    # Also include the start error file
    start_error = manager.last_start_error()

    return {
        "log_tail": log_text,
        "diagnoses": diagnoses,
        "start_error": start_error,
    }


@app.post("/api/server/start")
async def start_server(payload: StartPayload) -> dict[str, Any]:
    config = state.save_config(payload.config or state.get_config())
    status = manager.health(config)
    if status.get("managed") and status.get("healthy"):
        return {"status": "online", "server": status}
    # Run preflight checks — reject if critical checks fail
    preflight = run_preflight(config, log_path=manager.log_path)
    if not preflight["ready"]:
        issues = "; ".join(preflight["blocking_issues"])
        raise HTTPException(
            status_code=500,
            detail=f"Launch blocked: {issues}",
        )
    # Spawn llama.cpp in background — frontend polls /api/server/status
    import asyncio

    asyncio.get_event_loop().create_task(_bg_start(config))
    return {"status": "starting"}


@app.post("/api/server/stop")
def stop_server() -> dict[str, Any]:
    manager.stop()
    return {"stopped": True}


@app.post("/api/generation/stop")
def stop_generation() -> dict[str, Any]:
    return {"stopped": manager.stop_generation()}


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": scan_models(), "downloads": downloads.list_jobs()}


@app.post("/api/models/load")
async def load_model(payload: ModelLoadPayload) -> dict[str, Any]:
    config = state.get_config()
    config = apply_model_profile(config, payload.model_path)
    config = state.save_config(config)
    import asyncio

    asyncio.get_event_loop().create_task(_bg_start(config))
    return {"config": config, "status": "starting"}


@app.post("/api/models/download")
def start_download(payload: DownloadPayload) -> dict[str, Any]:
    try:
        job = downloads.start_download(payload.url, payload.destination_path)
        return {"job": job}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/models/download/{job_id}/cancel")
def cancel_download(job_id: str) -> dict[str, Any]:
    try:
        return {"job": downloads.cancel(job_id)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/chats")
def list_chats() -> dict[str, Any]:
    return {"chats": state.list_chats()}


@app.post("/api/chats")
def create_chat(payload: ChatCreatePayload) -> dict[str, Any]:
    return {"chat": state.create_chat(payload.title)}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int) -> dict[str, Any]:
    try:
        return {"chat": state.get_chat(chat_id)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int) -> dict[str, Any]:
    state.delete_chat(chat_id)
    return {"deleted": True}


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: int, payload: ChatRenamePayload) -> dict[str, Any]:
    try:
        state.rename_chat(chat_id, payload.title)
        return {"chat": state.get_chat(chat_id)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/chats/{chat_id}/messages")
def send_message(chat_id: int, payload: MessagePayload) -> dict[str, Any]:
    config = state.get_config()
    status = manager.health(config)
    if not status["healthy"]:
        raise HTTPException(status_code=409, detail="llama.cpp server is not running")

    chat = state.ensure_chat(chat_id)
    user_message = state.add_message(chat["chat_id"], "user", payload.content)
    state.rename_chat_if_placeholder(chat["chat_id"], payload.content)

    full_chat = state.get_chat(chat["chat_id"])
    messages: list[dict[str, str]] = []
    system_prompt = str(config.get("system_prompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in full_chat["messages"]:
        messages.append({"role": message["role"], "content": message["content"]})

    try:
        response = manager.chat(config, messages)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    assistant_message = None
    if response["content"]:
        assistant_message = state.add_message(chat["chat_id"], "assistant", response["content"])
    updated_chat = state.get_chat(chat["chat_id"])
    return {
        "chat": updated_chat,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "latency_ms": response["latency_ms"],
        "cancelled": response["cancelled"],
    }


@app.post("/api/chats/{chat_id}/messages/stream")
def stream_message(chat_id: int, payload: MessagePayload) -> StreamingResponse:
    config = state.get_config()
    status = manager.health(config)
    if not status["healthy"]:
        raise HTTPException(status_code=409, detail="llama.cpp server is not running")

    chat = state.ensure_chat(chat_id)
    user_message = state.add_message(chat["chat_id"], "user", payload.content)
    state.rename_chat_if_placeholder(chat["chat_id"], payload.content)
    full_chat = state.get_chat(chat["chat_id"])

    messages: list[dict[str, str]] = []
    system_prompt = str(config.get("system_prompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in full_chat["messages"]:
        messages.append({"role": message["role"], "content": message["content"]})

    def emit(event: dict[str, Any]) -> bytes:
        return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    def event_stream() -> Iterator[bytes]:
        yield emit(
            {
                "type": "start",
                "chat_id": chat["chat_id"],
                "user_message": user_message,
            }
        )
        try:
            for event in manager.chat_stream(config, messages):
                if event["type"] == "delta":
                    yield emit(event)
                    continue

                content = str(event.get("content") or "").strip()
                assistant_message = None
                if content:
                    assistant_message = state.add_message(chat["chat_id"], "assistant", content)
                updated_chat = state.get_chat(chat["chat_id"])
                yield emit(
                    {
                        "type": "done",
                        "chat": updated_chat,
                        "assistant_message": assistant_message,
                        "latency_ms": event["latency_ms"],
                        "cancelled": event["cancelled"],
                    }
                )
        except Exception as error:
            yield emit({"type": "error", "detail": str(error)})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ── Knowledge Base API ──────────────────────────────────────────


@app.get("/api/knowledge/stats")
def knowledge_stats() -> dict[str, Any]:
    return knowledge_db.stats()


@app.get("/api/knowledge/sources")
def knowledge_sources() -> dict[str, Any]:
    return {"sources": discover_sources(knowledge_db)}


@app.post("/api/knowledge/query")
def knowledge_query(payload: KnowledgeQueryPayload) -> dict[str, Any]:
    config = state.get_config()
    embedder = _embedder_from_config(config)
    results, search_stats = retrieve(
        query=payload.query,
        db=knowledge_db,
        embedder=embedder,
        top_k=payload.top_k,
        use_vectors=payload.use_vectors,
    )
    return {
        "results": [r.to_dict() for r in results],
        "stats": search_stats,
    }


@app.post("/api/knowledge/ingest")
def knowledge_ingest(payload: KnowledgeIngestPayload) -> dict[str, Any]:
    config = state.get_config()
    embedder = _embedder_from_config(config)
    results: list[dict[str, Any]] = []

    if payload.path:
        from pathlib import Path

        path = Path(payload.path).expanduser()
        if path.is_dir():
            ingest_results = ingest_directory(
                directory=path,
                source_type=payload.source,
                db=knowledge_db,
                embedder=embedder,
                pattern=payload.pattern,
                chunk_size=payload.chunk_size,
                embed=payload.embed,
            )
            results = [
                {
                    "source": r.source,
                    "path": r.path,
                    "records": r.records,
                    "chunks": r.chunks,
                    "embedded": r.embedded,
                    "errors": r.errors,
                }
                for r in ingest_results
            ]
        else:
            result = ingest_jsonl(
                path=path,
                source_type=payload.source,
                db=knowledge_db,
                embedder=embedder,
                chunk_size=payload.chunk_size,
                embed=payload.embed,
            )
            results = [
                {
                    "source": result.source,
                    "path": result.path,
                    "records": result.records,
                    "chunks": result.chunks,
                    "embedded": result.embedded,
                    "errors": result.errors,
                }
            ]
    else:
        from pathlib import Path

        from .knowledge.ingest import SOURCE_PATHS

        paths = SOURCE_PATHS.get(payload.source, [])
        for base_path in paths:
            if base_path.exists():
                ingest_results = ingest_directory(
                    directory=base_path,
                    source_type=payload.source,
                    db=knowledge_db,
                    embedder=embedder,
                    pattern=payload.pattern,
                    chunk_size=payload.chunk_size,
                    embed=payload.embed,
                )
                for r in ingest_results:
                    results.append(
                        {
                            "source": r.source,
                            "path": r.path,
                            "records": r.records,
                            "chunks": r.chunks,
                            "embedded": r.embedded,
                            "errors": r.errors,
                        }
                    )

    total_records = sum(r["records"] for r in results)
    total_chunks = sum(r["chunks"] for r in results)
    total_embedded = sum(r["embedded"] for r in results)
    all_errors = [e for r in results for e in r.get("errors", [])]

    return {
        "results": results,
        "summary": {
            "total_records": total_records,
            "total_chunks": total_chunks,
            "total_embedded": total_embedded,
            "errors": all_errors,
        },
    }


@app.post("/api/knowledge/embed")
def knowledge_embed(payload: KnowledgeEmbedPayload) -> dict[str, Any]:
    config = state.get_config()
    embedder = _embedder_from_config(config)
    if not embedder.is_available():
        raise HTTPException(
            status_code=409,
            detail="llama-server is not running — start it first to generate embeddings",
        )
    stats = embed_chunks_for_db(knowledge_db, embedder, batch_size=payload.batch_size)
    return stats


@app.delete("/api/knowledge")
def knowledge_clear() -> dict[str, Any]:
    knowledge_db.clear_all()
    return {"cleared": True}


def run() -> None:
    config = state.get_config()
    uvicorn.run(
        "llama_webui.main:app",
        host=str(config["bind_host"]),
        port=int(config["bind_port"]),
        reload=False,
    )
