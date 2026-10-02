"""
Local web GUI for the scorecard pipeline (#22).

    cd scorecard
    uv run streamlit run app.py

Runs on http://localhost:8501 only (see .streamlit/config.toml). It calls the
existing commands (scorecard.py) for anything that changes data, so the GUI
holds no pipeline logic of its own:
  Games  : list games, add a scan, read a scorecard (live log), view grid + widget
  Review : low-confidence plate appearances with their cell image; accept or
           correct -> updates _cells.json, then sync-edits + reimport-game
  Season : export Excel, publish
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import streamlit as st
import yaml

from db import get_data_root
from gamepaths import SCAN_EXTS, cells_json, find_scan, game_folder

HERE = Path(__file__).parent
RESULT_CODES = ["1B", "2B", "3B", "HR", "BB", "HP", "E", "FC", "K", "K-PB",
                "F7", "F8", "F9", "6-3", "4-3", "5-3", "SAC", "SF", "DP", "(other…)"]
_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}.+\((Home|Away)\)$")

st.set_page_config(page_title="Scorecards", page_icon="⚾", layout="wide")


# ── helpers ───────────────────────────────────────────────────────────────────

def review_threshold() -> int:
    cfg = yaml.safe_load((HERE / "config.yml").read_text(encoding="utf-8")) or {}
    return int((cfg.get("review") or {}).get("threshold", 2))


def list_games(root: Path) -> list[str]:
    names = {p.name for p in (root / "games").iterdir() if p.is_dir() and p.name[:4].isdigit()} \
        if (root / "games").exists() else set()
    if (root / "scans").exists():
        names |= {p.stem for p in (root / "scans").iterdir()
                  if p.suffix in SCAN_EXTS and p.stem[:4].isdigit()}
    return sorted(names)


def load_game(root: Path, name: str) -> dict | None:
    p = cells_json(root, name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def all_pas(game: dict) -> list[dict]:
    """Flatten to rows that remember where each PA lives in the JSON."""
    rows = []
    for si, slot in enumerate(game["lineup"]):
        for pi, pl in enumerate(slot["players"]):
            for ai, pa in enumerate(pl["plate_appearances"]):
                rows.append({"slot": slot["batting_order"], "player": pl["name"], "idx": (si, pi, ai), **pa})
    return rows


def run_command(args: list[str], root: Path) -> int:
    """Run `scorecard.py <args>` and stream its output into the page."""
    env = {**os.environ, "SCORECARD_DATA_ROOT": str(root), "PYTHONIOENCODING": "utf-8"}
    box = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen([sys.executable, "scorecard.py", *args], cwd=HERE, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]))
    proc.wait()
    box.code("\n".join(lines[-40:]) + f"\n\n[exit {proc.returncode}]")
    return proc.returncode


@st.cache_data(show_spinner="Locating the grid…")
def grid_for(root: str, name: str, scan_mtime: float):
    """Crop image + cell boundaries, detected the same way extract_cells does."""
    from probe_grid import detect_grid
    from rectify import detect_grid_lattice
    r, scan = Path(root), find_scan(Path(root), name)
    layout_p = game_folder(r, name) / "cells" / "_layout.json"
    layout = json.loads(layout_p.read_text()) if layout_p.exists() else {}
    a = layout.get("run_args", {})
    manual = any(a.get(k) is not None for k in ("grid_start", "grid_width", "cell_height"))
    res = None
    with contextlib.redirect_stdout(io.StringIO()):
        if not manual:
            res = detect_grid_lattice(str(scan), n_player_rows=a.get("n_player_rows") or 10,
                                      rectified_out=str(game_folder(r, name) / f"{name}_rectified.jpg"))
        if res is None:
            res = detect_grid(str(scan), n_player_rows=a.get("n_player_rows") or 10,
                              n_inning_cols=a.get("innings") or 9,
                              left_skip_frac=a.get("left_skip_frac") or 0.05,
                              grid_start=a.get("grid_start"), grid_col_width=a.get("grid_width"),
                              cell_height=a.get("cell_height"))
            res = (*res, str(scan), {})
    row_tops, row_bottoms, _, _, col_lefts, _, crop_path, _ = res
    return crop_path, row_tops, row_bottoms, col_lefts, layout.get("col_to_inning") or []


def cell_image(root: Path, name: str, game: dict, pa_row: dict):
    """The cache cell a PA came from: k-th non-empty cell of (row, inning) = k-th PA."""
    scan = find_scan(root, name)
    if scan is None:
        return None
    crop_path, tops, bottoms, lefts, c2i = grid_for(str(root), name, scan.stat().st_mtime)
    ri, inn = pa_row["slot"] - 1, pa_row["inning"]
    same = [r for r in all_pas(game) if r["slot"] == pa_row["slot"] and r["inning"] == inn]
    k = [r["idx"] for r in same].index(pa_row["idx"])
    cols = []
    for ci, ci_inn in enumerate(c2i):
        cf = game_folder(root, name) / "cells" / f"r{ri+1:02d}_c{ci+1:02d}.json"
        if ci_inn == inn and cf.exists() and json.loads(cf.read_text(encoding="utf-8")).get("result"):
            cols.append(ci)
    if k >= len(cols) or cols[k] + 1 >= len(lefts) or ri >= len(tops):
        return None
    img = cv2.imread(crop_path)
    ci = cols[k]
    crop = img[max(0, tops[ri]):bottoms[ri], max(0, lefts[ci]):lefts[ci + 1]]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def save_pa(root: Path, name: str, idx: tuple, result: str, run: bool) -> None:
    """Write a checked PA into _cells.json (confidence 5 = hand-checked), then push it
    to the cell cache and the DB with the existing commands."""
    p = cells_json(root, name)
    game = json.loads(p.read_text(encoding="utf-8"))
    si, pi, ai = idx
    pa = game["lineup"][si]["players"][pi]["plate_appearances"][ai]
    pa.update(result=result, run_scored=run, confidence=5, result_conf=5, run_conf=5,
              conf_reasons=["hand-checked"])
    p.write_text(json.dumps(game, ensure_ascii=False, indent=2), encoding="utf-8")
    run_command(["sync-edits", name], root)
    run_command(["reimport-game", name], root)


# ── pages ─────────────────────────────────────────────────────────────────────

root = Path(st.sidebar.text_input("Season folder", str(get_data_root())))
if not (root / "scans").exists():
    st.sidebar.error("No scans/ folder here.")
    st.stop()
page = st.sidebar.radio("Page", ["Games", "Review", "Season"])
games = list_games(root)
threshold = review_threshold()

if page == "Games":
    st.title("Games")
    rows = []
    for g in games:
        data = load_game(root, g)
        pas = all_pas(data) if data else []
        rows.append({"game": g, "read": "yes" if data else "—", "PAs": len(pas) or None,
                     "to review": sum(1 for r in pas if (r.get("confidence") or 5) <= threshold) or None})
    st.dataframe(rows, hide_index=True, width="stretch")

    with st.expander("Add a scan"):
        up = st.file_uploader("Scorecard image", type=["jpg", "jpeg", "png"])
        new_name = st.text_input("Game name", placeholder="2026-09-20 - Kinheim (Home)")
        if up and st.button("Save scan"):
            if not _NAME_RE.match(new_name.strip()):
                st.error("Name must look like: 2026-09-20 - Opponent (Home) or (Away)")
            else:
                dest = root / "scans" / f"{new_name.strip()}{Path(up.name).suffix.lower()}"
                dest.write_bytes(up.getvalue())
                st.success(f"Saved {dest.name}")
                st.rerun()

    game = st.selectbox("Game", games, index=len(games) - 1 if games else 0)
    if game:
        data = load_game(root, game)
        c1, c2, c3 = st.columns(3)
        reuse = c1.checkbox("Reuse cached cells", value=data is not None,
                            help="Only read cells that haven't been read before (no repeat API cost).")
        innings = c2.number_input("Innings (0 = remembered/default)", 0, 12, 0)
        if c3.button("Read scorecard", type="primary"):
            args = ["read-scorecard", game, "--yes"] + (["--reuse-cache"] if reuse else []) \
                + (["--innings", str(innings)] if innings else [])
            run_command(args, root)
            st.cache_data.clear()
        folder = game_folder(root, game)
        dbg = folder / f"{game}_grid_debug.png"
        widget = folder / f"{game}.html"
        tab1, tab2 = st.tabs(["Widget", "Grid"])
        with tab1:
            if widget.exists():
                st.iframe(widget, height=900)
            else:
                st.info("Not read yet.")
        with tab2:
            if dbg.exists():
                st.image(str(dbg), caption="Detected grid: red = rows, blue = columns")

elif page == "Review":
    st.title("Review")
    read = [g for g in games if cells_json(root, g).exists()]
    game = st.selectbox("Game", read, index=len(read) - 1 if read else 0)
    max_conf = st.slider("Show plate appearances with confidence at most", 1, 5, threshold)
    if game:
        data = load_game(root, game)
        todo = [r for r in all_pas(data) if (r.get("confidence") or 5) <= max_conf]
        st.caption(f"{len(todo)} to check. Accept or correct each; it's saved to the cache and DB right away.")
        for r in todo:
            key = f"{r['slot']}-{r['inning']}-{r['idx']}"
            with st.container(border=True):
                left, right = st.columns([1, 3])
                img = cell_image(root, game, data, r)
                if img is not None:
                    left.image(img, width=180)
                right.markdown(f"**Slot {r['slot']} · {r['player']} · inning {r['inning']}** — "
                               f"read as **{r['result']}**{' + run' if r['run_scored'] else ''} "
                               f"(confidence {r.get('confidence')})")
                right.caption("Why flagged: " + ("; ".join(r.get("conf_reasons") or []) or "—"))
                a, b, c, d = right.columns([2, 2, 1, 1])
                opts = RESULT_CODES if r["result"] in RESULT_CODES else [r["result"]] + RESULT_CODES
                choice = a.selectbox("Result", opts, index=opts.index(r["result"]), key=f"res{key}")
                if choice == "(other…)":
                    choice = b.text_input("Code", key=f"oth{key}").strip().upper()
                run = c.checkbox("Run", value=bool(r["run_scored"]), key=f"run{key}")
                if d.button("Save", key=f"save{key}", type="primary", disabled=not choice):
                    save_pa(root, game, r["idx"], choice, run)
                    st.rerun()

else:
    st.title("Season")
    if st.button("Export Excel workbook"):
        run_command(["export-excel"], root)
    dest = st.text_input("Publish to folder", placeholder=r"G:\My Drive\Quick 2026")
    if st.button("Publish widgets + workbook", disabled=not dest):
        run_command(["publish", str(root), dest], root)
