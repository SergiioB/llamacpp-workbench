from __future__ import annotations

import json
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .app_state import AppState
from .download_manager import ModelDownloadManager
from .llama_manager import LlamaServerManager
from .model_inventory import apply_model_profile, build_model_presets, list_candidate_models, scan_models
from .settings import PROJECT_ROOT, STATIC_DIR, data_dir, default_download_dir

state = AppState(PROJECT_ROOT)
manager = LlamaServerManager(state.log_path)
downloads = ModelDownloadManager(data_dir() / "downloads", default_download_dir())

app = FastAPI(title="llama-webui")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ConfigPayload(BaseModel):
    config: dict[str, Any]


class StartPayload(BaseModel):
    config: dict[str, Any] | None = None


class ChatCreatePayload(BaseModel):
    title: str | None = None


class MessagePayload(BaseModel):
    content: str


class DownloadPayload(BaseModel):
    url: str
    destination_path: str | None = None


class ModelLoadPayload(BaseModel):
    model_path: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


@app.post("/api/server/start")
def start_server(payload: StartPayload) -> dict[str, Any]:
    config = state.save_config(payload.config or state.get_config())
    try:
        status = manager.start(config)
        return {"status": status}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


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
def load_model(payload: ModelLoadPayload) -> dict[str, Any]:
    config = state.get_config()
    config = apply_model_profile(config, payload.model_path)
    config = state.save_config(config)
    try:
        status = manager.start(config)
        return {"config": config, "status": status}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


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
def stream_message(chat_id: int, payload: MessagePayload):
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

    def event_stream():
        yield emit({
            "type": "start",
            "chat_id": chat["chat_id"],
            "user_message": user_message,
        })
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
                yield emit({
                    "type": "done",
                    "chat": updated_chat,
                    "assistant_message": assistant_message,
                    "latency_ms": event["latency_ms"],
                    "cancelled": event["cancelled"],
                })
        except Exception as error:
            yield emit({"type": "error", "detail": str(error)})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def run() -> None:
    config = state.get_config()
    uvicorn.run(
        "llama_webui.main:app",
        host=str(config["bind_host"]),
        port=int(config["bind_port"]),
        reload=False,
    )
