from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def find_repo_root() -> Path:
    """Walk up from this file looking for pyproject.toml. Falls back to cwd."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def repo_path(*parts: str) -> Path:
    return find_repo_root().joinpath(*parts)
