# AGENTS.md

Instructions for AI agents working with the llama-webui codebase.

## Project Overview

llama-webui is a standalone local web interface for llama.cpp with persistent chats, model management, streaming responses, and hardware-aware runtime tuning. It runs on Python 3.11+ with FastAPI and Uvicorn.

## Quick Start

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install dev dependencies (for linting, testing)
uv pip install -e ".[dev]"

# Run the application
llama-webui
```

The web UI will be available at `http://localhost:8095`.

## Project Structure

```
llama-webui/
├── src/llama_webui/          # Main application code
│   ├── main.py              # FastAPI app and routes
│   ├── app_state.py         # SQLite database and state management
│   ├── llama_manager.py     # llama.cpp subprocess management
│   ├── model_inventory.py   # Model discovery and presets
│   ├── download_manager.py  # GGUF download handling
│   └── settings.py          # Path configuration
├── static/                   # Frontend (HTML, JS, CSS)
├── scripts/                  # Utility scripts (benchmarks, etc.)
├── docs/                     # Documentation
├── models/                   # GGUF model files (gitignored)
└── data/                     # Runtime database and logs (gitignored)
```

## Development Workflow

### Code Quality

```bash
# Run linter (ruff)
ruff check src/ scripts/

# Auto-fix linting issues
ruff check src/ scripts/ --fix

# Run formatter
ruff format src/ scripts/

# Run type checker (mypy)
mypy src/llama_webui/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing

```bash
# Run tests (when available)
pytest

# Run tests with coverage
pytest --cov=src/llama_webui --cov-report=term-missing
```

## Naming Conventions

### Python

- **Files**: `snake_case.py` (e.g., `app_state.py`, `llama_manager.py`)
- **Classes**: `PascalCase` (e.g., `AppState`, `LlamaServerManager`)
- **Functions**: `snake_case` (e.g., `get_config`, `start_server`)
- **Variables**: `snake_case` (e.g., `model_path`, `cpu_mask`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONFIG`, `THINK_TAG_PATTERN`)
- **Private methods**: Prefix with underscore (e.g., `_sanitize_visible_content`)
- **Type hints**: Always use type annotations for function parameters and return types

### FastAPI Routes

- Route functions should be named descriptively: `get_config`, `save_config`, `start_server`
- Pydantic models for request bodies: `*Payload` suffix (e.g., `ConfigPayload`, `StartPayload`)

### Database

- Tables: `snake_case` (e.g., `chats`, `messages`)
- Columns: `snake_case` (e.g., `chat_id`, `created_at`)

## Architecture

### Request Flow

1. User interacts with static frontend (`static/index.html`, `static/app.js`)
2. Frontend makes API calls to FastAPI routes in `main.py`
3. Routes delegate to managers (`LlamaServerManager`, `AppState`, `ModelDownloadManager`)
4. `LlamaServerManager` spawns and manages the llama.cpp subprocess
5. `AppState` persists configuration and chat history to SQLite

### Key Components

- **`AppState`**: SQLite-backed state management with thread-safe access
- **`LlamaServerManager`**: Subprocess lifecycle management for llama-server
- **`ModelDownloadManager`**: Async GGUF downloads with progress tracking
- **`model_inventory`**: Model discovery from configured directories

## Configuration

Environment variables (see `.env.example`):

- `LLAMA_WEBUI_DATA_DIR`: Runtime state directory (default: `./data`)
- `LLAMA_WEBUI_MODEL_DIRS`: Colon-separated model search paths
- `LLAMA_WEBUI_LLAMA_SERVER`: Path to llama-server binary
- `LLAMA_WEBUI_LLAMA_CLI`: Path to llama-cli binary

## RK3588-Specific Notes

For RK3588 boards, see `docs/rk3588-benchmarks.md` for tested configurations. Key presets:
- CPU affinity: `4-7` (big cluster)
- Threads: 4
- KV cache quantization: `--cache-type-k q8_0 --cache-type-v q4_0`

## Tech Debt Tracking

All TODO, FIXME, HACK, and XXX comments must link to GitHub issues for tracking.

### Accepted Formats

```python
# TODO(#123) - link to issue by number in current repo
# TODO(https://github.com/SergiioB/llamacpp-workbench/issues/123) - full URL
# FIXME(#456) - same patterns apply for FIXME
# HACK(#789) - same patterns apply for HACK
```

### Enforcement

- Ruff's FIX rules detect TODO/FIXME/HACK/XXX comments
- Pre-commit hook validates comments link to issues
- Run manually: `python scripts/check_tech_debt.py`

### Why This Matters

Linking tech debt to issues ensures:
- Work is tracked and prioritized
- Agents can find related context when fixing issues
- Technical debt doesn't accumulate silently

## Dependencies

Main dependencies:
- `fastapi>=0.115.0`: Web framework
- `uvicorn>=0.30.0`: ASGI server

Dev dependencies (optional):
- `ruff>=0.4.0`: Linting and formatting
- `mypy>=1.10.0`: Type checking
- `pytest>=8.0.0`: Testing
- `pytest-cov>=5.0.0`: Coverage

## Pre-commit Hooks

Install hooks with:
```bash
pre-commit install
```

Hooks run automatically before commits:
- ruff (linting + formatting + tech debt detection)
- mypy (type checking)
- check-added-large-files
- detect-private-key
- check-merge-conflict
- trailing-whitespace fix
- check-tech-debt (enforces TODO comments link to issues)
