#!/usr/bin/env python3
"""
Generate OpenAPI schema for llama-webui API.

This script exports the FastAPI OpenAPI schema to a JSON file for documentation
and client generation purposes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# ruff: noqa: E402
from llama_webui.main import app


def generate_openapi_schema() -> dict[str, Any]:
    """Generate OpenAPI schema from FastAPI app."""
    schema: dict[str, Any] = app.openapi()
    return schema


def save_schema(output_path: Path | None = None) -> Path:
    """Generate and save OpenAPI schema to file.

    Args:
        output_path: Path to save schema. Defaults to docs/openapi.json

    Returns:
        Path to saved schema file
    """
    if output_path is None:
        output_path = project_root / "docs" / "openapi.json"

    schema = generate_openapi_schema()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = save_schema()
    print(f"OpenAPI schema saved to: {output}")
