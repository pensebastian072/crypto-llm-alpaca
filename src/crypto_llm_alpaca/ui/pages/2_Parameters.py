from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml
import streamlit as st

from crypto_llm_alpaca._paths import find_repo_root
from crypto_llm_alpaca.env import load_env
from crypto_llm_alpaca.ui._runner import active_config_path

load_env()
st.set_page_config(page_title="Parameters", layout="wide")
st.title("Parameters")

DEFAULT = find_repo_root() / "config" / "default.yaml"
OVERRIDES = find_repo_root() / "config" / "ui_overrides.yaml"

if not DEFAULT.exists():
    st.error(f"`{DEFAULT}` not found. Run from the repo root.")
    st.stop()

with DEFAULT.open() as f:
    base_cfg = yaml.safe_load(f)

if OVERRIDES.exists():
    with OVERRIDES.open() as f:
        cur_cfg = yaml.safe_load(f) or base_cfg
    st.success(f"Overrides active: `{OVERRIDES}` (used by `{active_config_path()}`)")
else:
    cur_cfg = base_cfg
    st.info("No overrides yet. Saving will create `config/ui_overrides.yaml`.")

st.caption("Edit values below. Save writes `config/ui_overrides.yaml`. Default file is never modified.")

edited: dict = {k: dict(v) if isinstance(v, dict) else v for k, v in cur_cfg.items()}


def _edit_section(name: str, section: dict) -> dict:
    out: dict = {}
    with st.expander(name, expanded=name in ("strategy", "risk", "costs")):
        for key, val in section.items():
            wkey = f"{name}.{key}"
            if isinstance(val, bool):
                out[key] = st.checkbox(key, value=val, key=wkey)
            elif isinstance(val, int) and not isinstance(val, bool):
                out[key] = st.number_input(key, value=val, step=1, key=wkey)
            elif isinstance(val, float):
                out[key] = st.number_input(key, value=float(val), format="%.6f", key=wkey)
            elif isinstance(val, str):
                out[key] = st.text_input(key, value=val, key=wkey)
            elif isinstance(val, list):
                txt = st.text_area(
                    f"{key} (YAML list)",
                    value=yaml.safe_dump(val, sort_keys=False).strip(),
                    key=wkey,
                    height=120,
                )
                try:
                    out[key] = yaml.safe_load(txt)
                except yaml.YAMLError as exc:
                    st.error(f"{key}: {exc}")
                    out[key] = val
            else:
                out[key] = val
    return out


for section_name, section in cur_cfg.items():
    if isinstance(section, dict):
        edited[section_name] = _edit_section(section_name, section)

st.markdown("---")
cols = st.columns(3)
if cols[0].button("Save & reload", type="primary"):
    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    with OVERRIDES.open("w") as f:
        yaml.safe_dump(edited, f, sort_keys=False)
    st.success(f"Wrote {OVERRIDES}. Other UI pages will pick this up automatically.")
    st.cache_data.clear()
    st.rerun()

if cols[1].button("Discard overrides"):
    if OVERRIDES.exists():
        OVERRIDES.unlink()
        st.success("Removed overrides; default config is active again.")
        st.cache_data.clear()
        st.rerun()
    else:
        st.info("No overrides file to remove.")

with st.expander("Diff vs default", expanded=False):
    base_yaml = yaml.safe_dump(base_cfg, sort_keys=False)
    edited_yaml = yaml.safe_dump(edited, sort_keys=False)
    if base_yaml == edited_yaml:
        st.write("No changes vs default.")
    else:
        st.code(edited_yaml, language="yaml")
