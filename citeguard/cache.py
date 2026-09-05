"""Versioned local JSON response cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import CACHE_SCHEMA_VERSION


class FileCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def make_key(
        self,
        *,
        operation: str,
        provider: str,
        query: str,
        model: str = "",
        prompt_version: str = "",
    ) -> str:
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "operation": operation,
            "provider": provider,
            "query": query,
            "model": model,
            "prompt_version": prompt_version,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def get(self, key: str) -> Any | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
