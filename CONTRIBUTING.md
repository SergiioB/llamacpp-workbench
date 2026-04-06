# Contributing to llama-webui

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/SergiioB/llamacpp-workbench.git
cd llamacpp-workbench

# Create virtual environment
uv venv && source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run linting
ruff check .

# Run type checking
mypy src/

# Run tests
pytest
```

## Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints (enforced by mypy strict mode)
- Write docstrings for public functions

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run `ruff check . && mypy src/`
5. Commit with a clear message
6. Push and open a Pull Request

## Reporting Issues

Use the GitHub Issues tab. For bugs, please include:
- OS and Python version
- Hardware (especially if RK3588 or GPU)
- Steps to reproduce
- Expected vs actual behavior
