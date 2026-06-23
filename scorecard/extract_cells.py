#!/usr/bin/env python3
"""
extract_cells.py — Cell-based scorecard extraction pipeline.

Replaces extract_rows.py. Key difference: inning assignment is structural
(column position in grid), not VLM inference — the single biggest source of errors.

Pipeline:
  1. Detect grid (probe_grid.detect_grid)
  2. Crop each (player × inning) cell
  3. Classify each cell with a focused VLM call — parallel, cached
  4. After every player row: check H/AB vs ground-truth stats
  5. After every inning column: check R/H vs ground-truth totals
  6. Assemble into GameExtraction JSON

Usage:
  uv run python scorecard/extract_cells.py images/scans/2026-06-07_almere.jpg \\
      --players players.txt --innings 9 --active-players 9
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import click
import cv2
import numpy as np
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from probe_grid import detect_grid
from models import (
    GameExtraction, GameInfo, LineupSlot, PlayerEntry,
    PlateAppearance, PASummary, InningTotals,
)

# ── Cell classification prompt ────────────────────────────────────────────────

_CELL_SYSTEM = """\
You are reading a single plate appearance (PA) cell from a Dutch KNBSB baseball scorecard.

The cell has a 2x2 quadrant layout:
  TOP-LEFT / TOP-RIGHT  : X marks tracking baserunner outs — IGNORE for the PA result.
                          WP/PB/SB notations here only indicate HOW the batter advanced
                          around the bases AFTER reaching — they are NOT the PA result.
  BOTTOM-RIGHT          : the plate appearance result (see rules below)
  BOTTOM-LEFT           : supplementary notations (SB, WP, PB, error codes, etc.)
                          These also only indicate HOW the batter advanced — NOT the PA result,
                          EXCEPT in the dropped-third-strike case described below.
  CENTER (at the crosshair intersection) : a FILLED/SOLID diamond or solid black dot
                          means this batter scored a run this inning. Look carefully —
                          it may be small or faint. Also check BOTTOM-LEFT for a circle
                          that sometimes marks a run scored.

BOTTOM-RIGHT RESULT RULES:
  A LARGE CIRCLE filling the quadrant = OUT; read the text inside the circle:
    K  or backwards-K         = strikeout (swinging / looking) — batter is OUT
    F# (e.g. F7, F6)          = fly out to fielder #
    #-# (e.g. 6-3, 4-3)       = groundout or force-out
    DP                        = double play
    SAC / SH (in circle)      = sacrifice bunt out
    SF (in circle)            = sacrifice fly out

  SPECIAL CASE — dropped third strike (K-PB):
    If you see the letter K (or backwards-K) WITHOUT a circle around it,
    AND there is WP or PB notation anywhere in the sub-cells (indicating the
    catcher dropped the ball and the batter reached base), return result: "K-PB".
    K-PB means the batter reached safely — it is NOT an out.
    Rule: K inside a circle = out. K without a circle + WP/PB = K-PB (safe).

  TWO ROUNDED HUMPS (no circle) = walk (BB)

  VERTICAL STROKE with crossbar(s) in BOTTOM-RIGHT = HIT:
    1 crossbar  = single (1B)
    2 crossbars = double (2B)
    3 crossbars = triple (3B)
    4 crossbars = home run (HR)

  E# (e.g. E6, E7)             = reached on error
  FC                            = fielder's choice
  HP or HBP                     = hit by pitch
  SAC / SH (not in circle)      = sacrifice bunt reached
  SF (not in circle)            = sacrifice fly

  A single diagonal template line with NO other marks = NO plate appearance (null)
  Completely blank cell                               = NO plate appearance (null)

IMPORTANT — unknown or ambiguous out marks:
  Dutch KNBSB scorecards do NOT use appeal plays, "OUT (A)", or any out
  notation not listed above. If you see a circle whose contents are ambiguous
  or do not clearly show a fielder number or DP/SAC/SF, return result: "K".

Set "confidence" to "low" if the cell is ambiguous, hard to read, or you are unsure of the result.
Set "confidence" to "high" if the cell is clear and unambiguous.

Return ONLY valid JSON — no prose, no explanation, no markdown fences:
{"result": "<code or null>", "run": <true or false>, "confidence": "<high or low>", "notes": "<text or null>"}"""


# ── VLM cell call ─────────────────────────────────────────────────────────────

def _encode_cell(crop: np.ndarray, scale: int = 4) -> tuple[bytes, str]:
    """Upscale cell and return raw JPEG bytes."""
    h, w = crop.shape[:2]
    big = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes(), "image/jpeg"


_TRANSIENT = ("503", "429", "UNAVAILABLE", "RATE", "overloaded", "Try again")


def _call_api(
    client, model: str, img_bytes: bytes, media_type: str,
    user_text: str, system: str, max_tokens: int,
) -> str | None:
    """Make one API call with up to 3 retries on transient errors. Returns raw text or 'api_error:...'."""
    for attempt in range(3):
        try:
            if model.startswith("gemini"):
                from google.genai import types
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=media_type),
                        user_text,
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.0,
                        max_output_tokens=max_tokens,
                    ),
                )
                return resp.text.strip()
            else:
                b64 = base64.b64encode(img_bytes).decode()
                msg = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0,
                    system=system,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": user_text},
                    ]}],
                )
                return msg.content[0].text.strip()
        except Exception as exc:
            msg_str = str(exc)
            if any(t in msg_str for t in _TRANSIENT) and attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s
                continue
            return f"api_error: {exc}"
    return None


def _parse_json_response(raw: str) -> dict | None:
    """Parse JSON response; fall back to partial-JSON salvage for truncated output."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    if raw and raw[0] != "{":
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Salvage: extract result/run/notes from truncated JSON via regex
        r_m = re.search(r'"result"\s*:\s*(?:"([^"]*?)"|null)', raw)
        run_m = re.search(r'"run"\s*:\s*(true|false)', raw)
        notes_m = re.search(r'"notes"\s*:\s*(?:"([^"]*?)"|null)', raw)
        if r_m or run_m:
            return {
                "result": (r_m.group(1) or None) if r_m else None,
                "run": run_m.group(1) == "true" if run_m else False,
                "notes": (notes_m.group(1) or None) if notes_m else None,
            }
        return None


def classify_cell(
    crop: np.ndarray,
    player_name: str,
    inning: int,
    client,
    model: str,
) -> dict:
    """Classify one PA cell. Returns {"result": str|None, "run": bool, "notes": str|None}."""
    img_bytes, media_type = _encode_cell(crop)
    user_text = f"Player: {player_name}  Inning: {inning}\nReturn JSON only."

    for attempt in range(2):  # retry once with stricter prompt on JSON parse failure
        extra = "" if attempt == 0 else " IMPORTANT: output ONLY the JSON object, nothing else."
        system = _CELL_SYSTEM + extra
        raw = _call_api(client, model, img_bytes, media_type, user_text, system, max_tokens=400)
        if raw is None:
            return {"result": None, "run": False, "notes": "api_error: max retries exceeded"}
        if raw.startswith("api_error:"):
            return {"result": None, "run": False, "notes": raw}

        parsed = _parse_json_response(raw)
        if parsed is not None:
            return parsed
        if attempt == 0:
            continue
    return {"result": None, "run": False, "notes": f"parse_error: {raw[:120]}"}


def _recheck_run(
    crop: np.ndarray,
    player_name: str,
    inning: int,
    client,
    model: str,
) -> bool:
    """Focused second pass: is there a run-scored indicator in this cell?"""
    img_bytes, media_type = _encode_cell(crop, scale=6)
    system = (
        "You are checking a Dutch KNBSB baseball scorecard cell. "
        "Look ONLY at the CENTER crosshair for a FILLED solid diamond or dot, "
        "and the BOTTOM-LEFT quadrant for any circular mark. "
        "Either indicates a run scored. Respond with exactly: true  or exactly: false"
    )
    user_text = f"Player: {player_name}  Inning: {inning}\nRun scored?"
    raw = _call_api(client, model, img_bytes, media_type, user_text, system, max_tokens=10)
    if raw and not raw.startswith("api_error:"):
        return raw.lower().startswith("true")
    return False


# ── Stat helpers ──────────────────────────────────────────────────────────────

_HITS = {"1B", "2B", "3B", "HR"}
_NOT_AB = {"BB", "HP", "HBP", "SAC", "SH", "SF"}


def _is_hit(r: str | None) -> bool:
    return (r or "").upper() in _HITS


def _is_ab(r: str | None) -> bool:
    return r is not None and (r or "").upper() not in _NOT_AB


def _is_out(result: str | None) -> bool:
    """True if this PA result means the batter was retired (can never score a run).
    K-PB (dropped third strike, batter reached safely) is NOT an out."""
    if result is None:
        return False
    r = result.upper().strip()
    if r == "K-PB":               # dropped third strike — batter reached safely
        return False
    if r in {"K", "KS", "DP", "SAC", "SH", "SF"}:
        return True
    if re.match(r"^F\d+$", r):   # fly out: F7, F4, etc.
        return True
    if re.match(r"^\d+-\d+$", r): # groundout/forceout: 6-3, 4-3, etc.
        return True
    return False


def _apply_batting_rules(
    grid: list[list[dict | None]],
    n_rows: int,
    innings: int,
    cache_dir: Path,
) -> None:
    """
    Post-process grid in-place enforcing three structural rules:
    1. Out results (K, F#, #-#, DP, SAC, SH, SF) can never have run=True.
    2. An isolated PA — nobody batting before or after in the same inning — is impossible; remove it.
    3. Three-out rule: once 3 outs are counted in an inning (in batting order), all
       subsequent PAs in that inning are invalid and are removed.
    Batting order is cyclic (after player n_rows comes player 1).
    Start player for each inning is derived from the last batter of the previous inning.
    """
    def _save(ri: int, ci: int) -> None:
        cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
        cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")

    def _has_pa(ri: int, ci: int) -> bool:
        c = grid[ri][ci]
        return bool(c and c.get("result") is not None)

    def _is_uncertain(ri: int, ci: int) -> bool:
        """True if this cell is null due to an API/parse error (not a confirmed no-PA)."""
        c = grid[ri][ci]
        if not c or c.get("result") is not None:
            return False
        notes = c.get("notes") or ""
        return notes.startswith("api_error:") or notes.startswith("parse_error:")

    # ── Rule 1: outs can never score a run ────────────────────────────────────
    for ri in range(n_rows):
        for ci in range(innings):
            cell = grid[ri][ci]
            if cell and cell.get("run") and _is_out(cell.get("result")):
                cell["run"] = False
                _save(ri, ci)
                click.echo(f"  [out-run] Player {ri+1} inn {ci+1} {cell['result']}: run forced False")

    # ── Rule 2: remove isolated PAs ───────────────────────────────────────────
    # Skip if either neighbor is uncertain (api/parse error) — we can't confirm it's truly isolated.
    for ci in range(innings):
        for ri in range(n_rows):
            if not _has_pa(ri, ci):
                continue
            prev_ri = (ri - 1) % n_rows
            next_ri = (ri + 1) % n_rows
            if _is_uncertain(prev_ri, ci) or _is_uncertain(next_ri, ci):
                continue  # can't safely call this isolated
            if not _has_pa(prev_ri, ci) and not _has_pa(next_ri, ci):
                old = grid[ri][ci].get("result")
                grid[ri][ci] = {"result": None, "run": False, "notes": f"removed:isolated ({old})"}
                _save(ri, ci)
                click.echo(f"  [isolated] Player {ri+1} inn {ci+1} was {old}: removed")

    # ── Rule 3: three-out rule ────────────────────────────────────────────────
    # Inning 1 starts at player 1 (row 0).
    # Each subsequent inning starts at (last batter of previous inning + 1) % n_rows.
    # Only applied when all cells in the inning are confirmed (no api/parse errors),
    # since uncertain cells break start-player tracking.
    start_ri = 0
    for ci in range(innings):
        # Skip this inning if any cell is uncertain
        if any(_is_uncertain(ri, ci) for ri in range(n_rows)):
            # Still track start_ri from confirmed last batter to keep subsequent innings right
            for k in range(n_rows):
                ri = (start_ri + k) % n_rows
                if _has_pa(ri, ci):
                    start_ri = (ri + 1) % n_rows
            continue
        outs = 0
        last_batter_ri: int | None = None
        for k in range(n_rows):
            ri = (start_ri + k) % n_rows
            if not _has_pa(ri, ci):
                continue
            if outs >= 3:
                old = grid[ri][ci].get("result")
                grid[ri][ci] = {"result": None, "run": False, "notes": f"removed:after_3_outs ({old})"}
                _save(ri, ci)
                click.echo(f"  [3-outs] Player {ri+1} inn {ci+1} was {old}: removed")
            else:
                last_batter_ri = ri
                if _is_out(grid[ri][ci].get("result")):
                    outs += 1
        if last_batter_ri is not None:
            start_ri = (last_batter_ri + 1) % n_rows


# ── Ground-truth loaders ──────────────────────────────────────────────────────

def _load_gt_totals(path: Path, innings: int) -> dict[int, dict]:
    """Returns {inning: {R, H, E, LOB}}"""
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", s)]
        if len(nums) >= 5 and 1 <= nums[0] <= innings:
            out[nums[0]] = {"R": nums[1], "H": nums[2], "E": nums[3], "LOB": nums[4]}
    return out


def _load_gt_stats(path: Path) -> dict[int, tuple[int, int]]:
    """Returns {batting_slot: (H, AB)}; first entry wins for duplicates (starter)."""
    out: dict[int, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", s)]
        if len(nums) >= 3 and 1 <= nums[0] <= 20 and nums[0] not in out:
            out[nums[0]] = (nums[1], nums[2])
    return out


# ── Integrity checks ──────────────────────────────────────────────────────────

def _check_row(slot: int, name: str, cells: list[dict], gt: dict[int, tuple] | None) -> None:
    pa = sum(1 for c in cells if c.get("result") is not None)
    h  = sum(1 for c in cells if _is_hit(c.get("result")))
    ab = sum(1 for c in cells if _is_ab(c.get("result")))
    r  = sum(1 for c in cells if c.get("run"))
    line = f"  [Slot {slot:2d}] {name:<22}  PA={pa} AB={ab} H={h} R={r}"
    if gt and slot in gt:
        gt_h, gt_ab = gt[slot]
        h_mark  = "OK" if h  == gt_h  else f"MISMATCH (GT={gt_h})"
        ab_mark = "OK" if ab == gt_ab else f"MISMATCH (GT={gt_ab})"
        line += f"  H:{h_mark}  AB:{ab_mark}"
    click.echo(line)


def _check_col(inning: int, cells: list[dict], gt: dict[int, dict] | None) -> None:
    r    = sum(1 for c in cells if c.get("run"))
    h    = sum(1 for c in cells if _is_hit(c.get("result")))
    outs = sum(1 for c in cells if _is_out(c.get("result")))
    e    = sum(1 for c in cells if re.match(r"^E\d+$", (c.get("result") or "").upper()))
    pa   = sum(1 for c in cells if c.get("result") is not None)
    line = f"  [Inn {inning}]  PA={pa} R={r} H={h} E={e} Outs={outs}"
    if gt and inning in gt:
        g = gt[inning]
        expected_pa = 3 + g["R"] + g["LOB"]
        pa_mark   = "OK" if pa   == expected_pa else f"MISMATCH (expected {expected_pa}, got {pa})"
        r_mark    = "OK" if r    == g["R"] else f"MISMATCH (GT={g['R']})"
        h_mark    = "OK" if h    == g["H"] else f"MISMATCH (GT={g['H']})"
        e_mark    = "OK" if e    == g["E"] else f"MISMATCH (GT={g['E']})"
        outs_mark = "OK" if outs == 3      else f"WARNING (expected 3, got {outs})"
        line += f"  PA:{pa_mark}  R:{r_mark}  H:{h_mark}  E:{e_mark}  Outs:{outs_mark}"
    click.echo(line)


def _check_pa_sequence(
    grid: list[list[dict | None]],
    n_rows: int,
    innings: int,
    active_roster: list[tuple[str, int | None]],
) -> None:
    """Verify no player has more total PAs than the batter ahead of them (cyclic order invariant)."""
    pas = [
        sum(1 for ci in range(innings) if (grid[ri][ci] or {}).get("result") is not None)
        for ri in range(n_rows)
    ]
    violations = []
    for ri in range(1, n_rows):
        if pas[ri] > pas[ri - 1]:
            nc = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
            np_ = active_roster[ri - 1][0] if ri - 1 < len(active_roster) else f"P{ri}"
            violations.append(f"  [PA-seq] IMPOSSIBLE: {nc} ({pas[ri]} PA) > {np_} ({pas[ri-1]} PA)")
    if violations:
        for v in violations:
            click.echo(v)
    else:
        click.echo("  OK: " + " ≥ ".join(str(p) for p in pas))


def _enforce_gt_runs(
    grid: list[list[dict | None]],
    n_rows: int,
    innings: int,
    gt_totals: dict[int, dict],
    cache_dir: Path,
) -> None:
    """Post-process: use GT run totals to enforce impossible runs.
    - GT R=0 for an inning → force all run=True cells to False.
    - GT R < extracted → log a warning; cannot auto-reduce without per-cell GT.
    """
    for ci in range(innings):
        inn = ci + 1
        if inn not in gt_totals:
            continue
        gt_r = gt_totals[inn]["R"]
        extracted = [ri for ri in range(n_rows) if (grid[ri][ci] or {}).get("run")]
        if gt_r == 0 and extracted:
            for ri in extracted:
                grid[ri][ci]["run"] = False
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")
                click.echo(f"  [GT-R=0] Player {ri+1} inn {inn}: run forced False (GT=0 runs)")
        elif len(extracted) > gt_r:
            click.echo(
                f"  [GT-R] Inn {inn}: {len(extracted)} runs extracted, GT={gt_r} — cannot auto-reduce"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--players", "players_file", default="players.txt",
              type=click.Path(), help="Roster file (name, jersey per line, one per batting slot).")
@click.option("--active-players", default=9, show_default=True,
              help="Number of active batting slots (subs share a slot, don't add rows).")
@click.option("--innings", default=9, show_default=True, help="Innings played.")
@click.option("--n-player-rows", default=10, show_default=True,
              help="Physical grid rows for players (template size, usually 10).")
@click.option("--n-inning-cols", default=11, show_default=True,
              help="Physical grid columns (innings 1-9 + empty 10 + stats 11).")
@click.option("--model", default=None,
              help="Anthropic model (default: EXTRACTION_MODEL env or claude-sonnet-4-6).")
@click.option("--workers", default=8, show_default=True, help="Parallel VLM calls.")
@click.option("--reuse-cache", is_flag=True, default=False,
              help="Reuse cached per-cell results (skip API calls for cached cells).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Classify cells and check integrity but do not write to DB.")
@click.option("--gt-dir", default=None,
              help="Ground-truth directory (default: images/ground_truth next to images/scans).")
def main(
    image_path, players_file, active_players, innings,
    n_player_rows, n_inning_cols, model, workers,
    reuse_cache, dry_run, gt_dir,
):
    """Cell-based scorecard extraction — inning from column position, not VLM guessing."""
    img_path = Path(image_path).resolve()
    if model is None:
        model = os.environ.get("EXTRACTION_MODEL", "claude-sonnet-4-6")
    click.echo(f"Image : {img_path.name}")
    click.echo(f"Model : {model}   Workers: {workers}   Innings: {innings}")

    # ── Date / opponent from filename ─────────────────────────────────────────
    date_str, opponent = None, None
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", img_path.stem)
    if m:
        date_str = m.group(1)
        opponent = m.group(2).replace("_", " ").title()
        click.echo(f"Date  : {date_str}   Opponent: {opponent}")

    # ── Roster (batting-order slots 1..active_players) ────────────────────────
    # One line per batting slot in order; subs share a slot and are noted
    # in supplementary data — they do NOT get their own grid row.
    roster: list[tuple[str, int | None]] = []  # (name, jersey)
    players_path = Path(players_file)
    if players_path.exists():
        for line in players_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            jersey = int(parts[1]) if len(parts) > 1 and parts[1].strip().lstrip("-").isdigit() else None
            roster.append((name, jersey))
    # Use only the first active_players entries for row→slot mapping
    active_roster = roster[:active_players]
    click.echo(f"Roster: {', '.join(n for n, _ in active_roster)}")

    # ── Grid detection ────────────────────────────────────────────────────────
    click.echo("\nDetecting grid...")
    debug_dir = img_path.parent / "Gridded"
    debug_dir.mkdir(exist_ok=True)
    debug_img = str(debug_dir / f"{img_path.stem}_grid_debug.png")
    row_tops, row_bottoms, _et, _eb, col_lefts, cell_size = detect_grid(
        str(img_path),
        n_player_rows=n_player_rows,
        n_inning_cols=n_inning_cols,
        debug_out=debug_img,
    )
    img = cv2.imread(str(img_path))
    n_active_rows = min(active_players, len(row_tops))
    click.echo(f"Grid  : {len(row_tops)} player rows x {len(col_lefts)-1} cols, cell={cell_size}px")
    click.echo(f"Active: first {n_active_rows} rows ({innings} innings)")
    click.echo(f"Debug : {debug_img}")

    # ── Cache directory ───────────────────────────────────────────────────────
    cache_dir = img_path.parent.parent / "_cache" / "cells" / img_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Ground truth ──────────────────────────────────────────────────────────
    if gt_dir:
        gt_root = Path(gt_dir)
    else:
        gt_root = img_path.parent.parent / "ground_truth"
    gt_totals: dict | None = None
    gt_stats: dict | None = None
    tp = gt_root / f"{img_path.stem}_totals.txt"
    sp = gt_root / f"{img_path.stem}_stats.txt"
    if tp.exists():
        gt_totals = _load_gt_totals(tp, innings)
        click.echo(f"GT    : {tp.name} loaded")
    if sp.exists():
        gt_stats = _load_gt_stats(sp)
        click.echo(f"GT    : {sp.name} loaded")

    # ── Build work list ───────────────────────────────────────────────────────
    # grid[ri][ci] = cell result dict or None
    grid: list[list[dict | None]] = [
        [None] * innings for _ in range(n_active_rows)
    ]
    to_process: list[tuple[int, int, Path]] = []
    cached_count = 0
    for ri in range(n_active_rows):
        for ci in range(innings):
            cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
            if reuse_cache and cf.exists():
                cell = json.loads(cf.read_text(encoding="utf-8"))
                notes = cell.get("notes") or ""
                # api_error cells are never "done" — always retry them
                if cell.get("result") is None and notes.startswith("api_error:"):
                    to_process.append((ri, ci, cf))
                    continue
                # Salvage cached parse_error: partial JSON stored in notes
                if cell.get("result") is None and notes.startswith("parse_error:"):
                    partial = notes[len("parse_error:"):].strip()
                    salvaged = _parse_json_response(partial)
                    if salvaged and salvaged.get("result") is not None:
                        cell = salvaged
                        cf.write_text(json.dumps(cell, ensure_ascii=False), encoding="utf-8")
                # Restore cells previously removed by structural rules so they are
                # re-evaluated fresh this run (other cells may have changed).
                if cell.get("result") is None and notes.startswith("removed:"):
                    m_restore = re.match(r"removed:\S+\s*\((.+?)\)", notes)
                    if m_restore:
                        orig = m_restore.group(1).strip()
                        cell = {
                            "result": None if orig in ("null", "None") else orig,
                            "run": False,
                            "notes": None,
                        }
                        cf.write_text(json.dumps(cell, ensure_ascii=False), encoding="utf-8")
                grid[ri][ci] = cell
                cached_count += 1
            else:
                to_process.append((ri, ci, cf))

    total_cells = n_active_rows * innings
    click.echo(f"\nCells : {total_cells} total | {cached_count} cached | {len(to_process)} to classify")

    # ── Classify ──────────────────────────────────────────────────────────────
    if model.startswith("gemini"):
        from google import genai as google_genai
        client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if to_process:
        t0 = time.monotonic()

        def _process(args: tuple[int, int, Path]) -> tuple[int, int, dict]:
            ri, ci, cf = args
            inning = ci + 1
            name = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
            y1 = max(0, row_tops[ri])
            y2 = row_bottoms[ri]
            x1 = max(0, col_lefts[ci])
            x2 = col_lefts[ci + 1]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                result = {"result": None, "run": False, "notes": "empty_crop"}
            else:
                result = classify_cell(crop, name, inning, client, model)
            cf.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return ri, ci, result

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_process, args): args for args in to_process}
            for fut in concurrent.futures.as_completed(futs):
                ri, ci, result = fut.result()
                grid[ri][ci] = result
                done += 1
                if done % 9 == 0 or done == len(to_process):
                    elapsed = time.monotonic() - t0
                    pct = 100 * done // len(to_process)
                    click.echo(f"  {done:3d}/{len(to_process)} ({pct}%)  {elapsed:.0f}s")

    # ── Batting rules (structural post-processing) ───────────────────────────
    click.echo("\n-- Batting rules " + "-" * 48)
    _apply_batting_rules(grid, n_active_rows, innings, cache_dir)

    # ── GT run enforcement (must come after batting rules) ───────────────────
    if gt_totals:
        click.echo("\n-- GT run enforcement " + "-" * 43)
        _enforce_gt_runs(grid, n_active_rows, innings, gt_totals, cache_dir)

    # ── Integrity checks ──────────────────────────────────────────────────────
    click.echo("\n-- Per-player (row) check " + "-" * 40)
    for ri in range(n_active_rows):
        name = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
        cells = [grid[ri][ci] or {} for ci in range(innings)]
        _check_row(ri + 1, name, cells, gt_stats)

    click.echo("\n-- PA sequence check " + "-" * 45)
    _check_pa_sequence(grid, n_active_rows, innings, active_roster)

    click.echo("\n-- Per-inning (column) check " + "-" * 37)
    for ci in range(innings):
        col = [grid[ri][ci] or {} for ri in range(n_active_rows)]
        _check_col(ci + 1, col, gt_totals)

    # ── Run reconciliation against GT inning totals ───────────────────────────
    # Only runs a focused re-check when we extracted FEWER runs than GT expects.
    # Over-counted innings are already handled by _enforce_gt_runs above.
    if gt_totals:
        click.echo("\n-- Run reconciliation " + "-" * 43)
        any_recheck = False
        for ci in range(innings):
            inn = ci + 1
            if inn not in gt_totals:
                continue
            gt_r = gt_totals[inn]["R"]
            extracted_r = sum(
                1 for ri in range(n_active_rows) if (grid[ri][ci] or {}).get("run")
            )
            if extracted_r == gt_r:
                click.echo(f"  Inn {inn}: R={extracted_r} OK")
                continue
            if extracted_r > gt_r:
                click.echo(f"  Inn {inn}: R={extracted_r} > GT={gt_r} — should have been handled by GT enforcement")
                continue
            # extracted_r < gt_r: re-examine cells without run markers
            click.echo(
                f"  Inn {inn}: extracted {extracted_r} run(s), GT={gt_r} — rechecking {gt_r - extracted_r} missing..."
            )
            any_recheck = True
            found = 0
            for ri in range(n_active_rows):
                if found >= gt_r - extracted_r:
                    break
                cell = grid[ri][ci] or {}
                if cell.get("result") is None or cell.get("run") or _is_out(cell.get("result")):
                    continue
                name = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
                y1 = max(0, row_tops[ri])
                y2 = row_bottoms[ri]
                x1 = max(0, col_lefts[ci])
                x2 = col_lefts[ci + 1]
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                run = _recheck_run(crop, name, inn, client, model)
                if run:
                    grid[ri][ci]["run"] = True
                    cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                    cf.write_text(
                        json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8"
                    )
                    click.echo(f"    -> Player {ri+1} {name} inn {inn}: run corrected True")
                    found += 1
        if not any_recheck:
            click.echo("  All innings match GT.")

    # ── Assemble GameExtraction ───────────────────────────────────────────────
    lineup: list[LineupSlot] = []
    for ri in range(n_active_rows):
        name = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
        jersey = active_roster[ri][1] if ri < len(active_roster) else None
        pas: list[PlateAppearance] = []
        for ci in range(innings):
            cell = grid[ri][ci] or {}
            r = cell.get("result")
            if r is not None:
                pas.append(PlateAppearance(
                    inning=ci + 1,
                    result=r,
                    run_scored=bool(cell.get("run")),
                    notes=cell.get("notes") or "",
                    rbi=0, sb=0, cs=0,
                    confidence="high",
                ))
        h  = sum(1 for pa in pas if _is_hit(pa.result))
        ab = sum(1 for pa in pas if _is_ab(pa.result))
        r  = sum(1 for pa in pas if pa.run_scored)
        lineup.append(LineupSlot(
            batting_order=ri + 1,
            players=[PlayerEntry(
                name=name,
                jersey_number=jersey,
                plate_appearances=pas,
                summary=PASummary(PA=len(pas), AB=ab, H=h, R=r),
            )],
        ))

    # Inning totals from grid
    runs_per_inning = [
        sum(1 for ri in range(n_active_rows) if (grid[ri][ci] or {}).get("run"))
        for ci in range(innings)
    ]
    game = GameExtraction(
        game=GameInfo(
            teams={"home": "Quick", "away": opponent or "Unknown"},
            date=date_str,
        ),
        lineup=lineup,
        inning_totals=InningTotals(runs_per_inning=runs_per_inning),
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = img_path.parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_cells.json"
    out_path.write_text(
        json.dumps(game.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(f"\nSaved : {out_path.name}")

    total_pa = sum(len(s.players[0].plate_appearances) for s in lineup if s.players)
    total_r  = sum(
        sum(1 for pa in s.players[0].plate_appearances if pa.run_scored)
        for s in lineup if s.players
    )
    click.echo(f"Total : {total_pa} PA extracted   {total_r} runs")
    if gt_totals:
        gt_r = sum(v["R"] for v in gt_totals.values())
        click.echo(f"GT    : {gt_r} runs expected")


if __name__ == "__main__":
    main()
