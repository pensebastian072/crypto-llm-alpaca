from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .._paths import find_repo_root


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def active_config_path(default: str = "config/default.yaml") -> str:
    """Return the active dashboard config path."""
    root = find_repo_root()
    env_config = os.environ.get("CRYPTO_LLM_CONFIG")
    if env_config:
        env_path = Path(env_config)
        return str(env_path if env_path.is_absolute() else root / env_path)
    overrides = root / "config" / "ui_overrides.yaml"
    if overrides.exists():
        return str(overrides)
    default_path = Path(default)
    return str(default_path if default_path.is_absolute() else root / default_path)


def run_cli(args: list[str], *, config: str | None = None, timeout: float | None = None) -> CliResult:
    """Invoke the crypto-llm CLI in a subprocess. Returns captured output."""
    cmd = [sys.executable, "-m", "crypto_llm_alpaca.cli"]
    if config is not None:
        cmd += ["--config", config]
    else:
        cmd += ["--config", active_config_path()]
    cmd += args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=str(find_repo_root()),
        timeout=timeout,
        check=False,
    )
    return CliResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


_ERROR_LINE = re.compile(r"^[A-Za-z_.]*(Error|Exception)\b|^error\b", re.IGNORECASE)
_MISSING_KEY = re.compile(
    r"(api[_ ]?key|token|secret)\w*.*\b(required|missing|not set)"
    r"|(required|missing|not set)\b.*\b(api[_ ]?key|token)"
    r"|KeyError: '[A-Z0-9_]*(KEY|TOKEN|SECRET)[A-Z0-9_]*'",
    re.IGNORECASE,
)


def error_summary(stderr: str) -> str:
    """The one line of a traceback a person needs: the final exception message."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if _ERROR_LINE.match(line):
            return line
    return lines[-1] if lines else "no error output"


def render_result(st_module, result: CliResult, label: str = "CLI output") -> None:
    """Render a CliResult; on failure show the exception line up front and keep the raw
    output in a collapsed expander. Clear cache + rerun on success."""
    badge = "OK" if result.ok else f"FAIL ({result.returncode})"
    if not result.ok:
        summary = error_summary(result.stderr)
        st_module.error(f"**{label} failed:** {summary}")
        if _MISSING_KEY.search(summary):
            st_module.info("Add the missing key to `.env` (see `.env.example`), then restart the app.")
    with st_module.expander(f"{label} — {badge}", expanded=False):
        if result.stdout:
            st_module.code(result.stdout, language="json")
        if result.stderr:
            st_module.code(result.stderr, language="text")
    if result.ok:
        st_module.cache_data.clear()
        st_module.rerun()
