| name | description |
|------|-------------|
| llama-server-integration | Patterns for integrating with llama.cpp server subprocess |

## llama.cpp Server Integration

Use this skill when working with `llama_manager.py` or any code that interacts with the llama.cpp server subprocess.

### Subprocess Management

- Use `subprocess.Popen` for non-blocking server startup
- Capture stdout/stderr to log files for debugging
- Track process PID for lifecycle management

### Health Checking

- Implement HTTP-based health checks to the llama.cpp server
- Check if the process is still alive before HTTP checks
- Return structured status: `{"healthy": bool, "pid": int | None, ...}`

### Configuration Mapping

Map llama-webui config keys to llama-server CLI flags:

```python
flag_map = {
    "ctx_size": "--ctx-size",
    "threads": "--threads",
    "gpu_layers": "--n-gpu-layers",
    # etc.
}
```

### Streaming Chat

- Use `requests.post(..., stream=True)` for streaming completions
- Parse Server-Sent Events (SSE) format from llama.cpp
- Yield structured events with `type`, `content`, and metadata

### Cleanup

- Always terminate subprocesses gracefully on shutdown
- Clean up stale processes on startup
- Handle SIGTERM/SIGINT signals properly
