from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.ui._runner import active_config_path, run_cli

load_env()
st.set_page_config(page_title="Doctor", layout="wide")
st.title("Doctor — preflight checks")

config_path = active_config_path()
st.caption(f"Config: `{config_path}`")

if st.button("Re-check", type="primary"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Running checks ..."):
    result = run_cli(["doctor", "--json"], config=config_path, timeout=60)

if not result.stdout:
    st.error("doctor produced no output")
    if result.stderr:
        st.code(result.stderr)
    st.stop()

try:
    payload = json.loads(result.stdout)
except json.JSONDecodeError:
    st.error("doctor output was not JSON")
    st.code(result.stdout)
    st.stop()

checks = payload.get("checks", [])
badge = {"ok": "✅", "warn": "⚠️", "fail": "❌"}
for c in checks:
    icon = badge.get(c["status"], "?")
    st.markdown(f"{icon} **{c['name']}** — {c['detail']}")

failed = [c for c in checks if c["status"] == "fail"]
warned = [c for c in checks if c["status"] == "warn"]
st.markdown("---")
st.write(f"**{len(checks)} checks** — {len(failed)} fail, {len(warned)} warn, {len(checks) - len(failed) - len(warned)} ok")
if failed:
    st.error("Fix the failed checks before running paper trading.")
elif warned:
    st.info("Warnings won't block backtesting/paper trading but signal something to look at.")
else:
    st.success("All clear.")
