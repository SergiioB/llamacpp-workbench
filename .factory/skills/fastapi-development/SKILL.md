| name | description |
|------|-------------|
| fastapi-development | Guidelines for developing FastAPI routes and handlers in llama-webui |

## FastAPI Development

Use this skill when working on FastAPI routes, request/response models, or API endpoints in the llama-webui project.

### Route Conventions

- Route functions should be named descriptively: `get_config`, `save_config`, `start_server`
- Use type hints for all parameters and return values
- Pydantic models for request bodies should use the `*Payload` suffix (e.g., `ConfigPayload`, `StartPayload`)
- Return `dict[str, Any]` for JSON responses

### Error Handling

- Use `HTTPException` from FastAPI for error responses
- Always include meaningful error messages
- Use appropriate status codes: 404 for not found, 409 for conflicts, 500 for server errors

### Request Models

```python
from pydantic import BaseModel

class ConfigPayload(BaseModel):
    config: dict[str, Any]
```

### Streaming Responses

- Use `StreamingResponse` from FastAPI for streaming endpoints
- Set appropriate media types (e.g., `application/x-ndjson` for newline-delimited JSON)

### Example Route

```python
@app.post("/api/server/start")
def start_server(payload: StartPayload) -> dict[str, Any]:
    config = state.save_config(payload.config or state.get_config())
    try:
        status = manager.start(config)
        return {"status": status}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
```
