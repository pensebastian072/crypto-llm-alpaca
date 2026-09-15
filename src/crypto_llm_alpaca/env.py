from __future__ import annotations

from pathlib import Path

from ._paths import find_repo_root

try:
    from dotenv import load_dotenv
except ImportError:  # graceful fallback if dep not installed yet
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


def load_env(path: Path | str | None = None) -> bool:
    """Load .env from repo root (or given path) without overriding existing process env."""
    target = Path(path) if path else find_repo_root() / ".env"
    if target.exists():
        load_dotenv(target, override=False)
    return target.exists()


def reset_for_tests() -> None:
    pass
