from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ._paths import find_repo_root


def _cache_dir(cache_dir: str | Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else find_repo_root() / "data" / "cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "cache"


def get_cache_path(key: str, cache_dir: str | Path | None = None) -> Path:
    return _cache_dir(cache_dir) / f"{_safe_key(key)}.json"


def load_cache(key: str, max_age_sec: int, cache_dir: str | Path | None = None) -> Any | None:
    path = get_cache_path(key, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    ts = float(data.get("ts", 0.0) or 0.0)
    if max_age_sec >= 0 and time.time() - ts > max_age_sec:
        return None
    return data.get("value")


def save_cache(key: str, value: Any, cache_dir: str | Path | None = None) -> Path:
    path = get_cache_path(key, cache_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ts": time.time(), "value": value}, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
