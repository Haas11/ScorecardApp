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
  uv run python extract_cells.py "Quick 2026 data/scans/2026-06-07 - Almere (Away).jpg"
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

The cell has a 2x2 quadrant layout (mimicking 3 bases & homeplate):
  BOTTOM-RIGHT (1st base)          : the plate appearance result (see rules below)
  TOP-LEFT (3rd) / TOP-RIGHT (2nd) : X marks track extra bases in 2B, 3B & HR — IGNORE for the PA result.
                                     WP/PB/SB notations here only indicate HOW the batter advanced
                                     around the bases AFTER reaching — they are NOT the PA result.  
  BOTTOM-LEFT (homeplate)          : supplementary notations (SB, WP, PB, error codes, etc.)
                                    These also only indicate HOW the batter advanced — NOT the PA result,
                                    if anything, record a run!                                    
                                    EXCEPT in the dropped-third-strike case described below.
                                    
  CENTER (at or near the crosshair intersection) : a FILLED/SOLID diamond or solid black dot
                          means this batter scored a run this inning. Look carefully —
                          it may be small or faint or slightly off center. 
                          Also check BOTTOM-LEFT for anything, this marks a run scored.
                          If a run is recorded, the PA result can not also be an out.

BOTTOM-RIGHT RESULT RULES:
  A LARGE CIRCLE filling the quadrant = OUT; 
  a PA out is ONLY EVER recorded with a LARGE CIRCLE, NO CIRCLE -> NEVER AN OUT!
  read the text inside the circle:
    K  or backwards-K         = strikeout (swinging / looking) — batter is OUT
    F# (e.g. F7, F6)          = fly out to fielder #
    #-# (e.g. 6-3, 4-3)       = groundout or force-out
    DP                        = double play
    SAC / SH (in circle)      = sacrifice bunt out
    SF (in circle)            = sacrifice fly out
    if multiple symbols inside the circle, or unambiguous, record a F out, not a K out.
    
    a smaller circle inside one of the subcells means a runner got out
    while running the bases, not during its PA. So it does count towards an out that inning, 
    but it does NOT define its PA. It's PA result is in the BOTTOM-RIGHT.    

  SPECIAL CASE — dropped third strike (K-PB):
    If you see the letter K (or backwards-K) WITHOUT a circle around it,
    AND there is WP or PB notation anywhere in the sub-cells (indicating the
    catcher dropped the ball and the batter reached base), return result: "K-PB".
    K-PB means the batter reached safely — it is NOT an out.
    Rule: K inside a circle = out. K without a circle + WP/PB = K-PB (safe).

    NO CIRCLE -> NEVER AN OUT!

  TWO ROUNDED HUMPS (no circle) = walk (BB)

  VERTICAL STROKE with horizontal crossbar(s) in BOTTOM-RIGHT = HIT:
    1 crossbar or "i" written = single (1B) (can look like L or reversed L )
    2 crossbars = double (2B)
    3 crossbars = triple (3B)
    4 crossbars = home run (HR) 
    HR crossbars will cover all 4 cells (and has a center run dot). 

    The crossbars are HORIZONTAL lines through the vertical stroke.
    A 2B or 3B can look like the letter K (same shape of strokes).
    DECISION RULE: if there is NO WP or PB notation anywhere in the cell,
    it CANNOT be K-PB — return 2B or 3B (count the crossbars carefully).

  E# (e.g. E6, E7)              = reached on error (NEVER an out)
  FC                            = fielder's choice
  HP or HBP                     = hit by pitch

  A single diagonal template line with NO other marks = NO plate appearance
  Completely blank cell                               = NO plate appearance

IMPORTANT — unknown or ambiguous out marks:
  Dutch KNBSB scorecards do NOT use appeal plays, "OUT (A)", or any out
  notation not listed above. If you see a circle whose contents are ambiguous
  or do not clearly show a fielder number or DP/SAC/SF, return result: "K".

Set "confidence" to "low" if the cell is ambiguous, hard to read, or you are unsure of the result.
Set "confidence" to "high" if the cell is clear and unambiguous.

Return ONLY valid JSON — no prose, no explanation, no markdown fences.
For a PA result use the string code. For no PA use JSON null (NOT the string "null").
PA cell:    {"result": "K", "run": false, "confidence": "high", "notes": null}
Empty cell: {"result": null, "run": false, "confidence": "high", "notes": null}"""


# ── VLM cell call ─────────────────────────────────────────────────────────────

def _detect_wrap_from_grid(
    grid: list[list[dict | None]],
    n_rows: int,
    innings: int,
) -> tuple[list[int], list[int]]:
    """
    Detect inning wrapping from raw VLM results (called AFTER cell classification,
    BEFORE batting rules so no cells have been removed yet).

    A column is "full" (inning wraps into next column) when:
    - All n_rows cells have a non-null result (every batter in the lineup batted), AND
    - Fewer than 3 outs in that column (the inning wasn't over yet).

    Returns:
      col_to_inning  — 1-based inning index for each column (length = innings)
      overflow_cols  — 1-based column indices that are overflow columns (same inning
                       as the column immediately to their left)
    """
    col_to_inning: list[int] = []
    overflow_cols: list[int] = []
    inning_num = 1

    for ci in range(innings):
        col_to_inning.append(inning_num)
        non_null = sum(
            1 for ri in range(n_rows)
            if (grid[ri][ci] or {}).get("result") is not None
        )
        outs = sum(
            1 for ri in range(n_rows)
            if _is_out((grid[ri][ci] or {}).get("result"))
        )
        if non_null == n_rows and outs < 3 and ci + 1 < innings:
            # All batters went to plate but < 3 outs → inning continues into next column
            overflow_cols.append(ci + 2)  # 1-based column index
        else:
            inning_num += 1

    return col_to_inning, overflow_cols


def _encode_cell(crop: np.ndarray, scale: int = 4) -> tuple[bytes, str]:
    """Upscale cell and return raw JPEG bytes."""
    h, w = crop.shape[:2]
    big = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes(), "image/jpeg"


def _encode_name_strip(crop: np.ndarray, target_h: int = 200) -> tuple[bytes, str]:
    """Encode a name-strip sub-row for VLM reading.

    Scales height to target_h while keeping the original width — the strip is
    already wide enough; only the height needs upscaling so text is legible.
    Uses PNG (lossless) to avoid JPEG block artifacts on thin handwritten strokes.
    """
    h, w = crop.shape[:2]
    new_h = max(target_h, h)
    big = cv2.resize(crop, (w, new_h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", big)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes(), "image/png"


_TRANSIENT = ("503", "429", "UNAVAILABLE", "RATE", "overloaded", "Try again", "RESOURCE_EXHAUSTED")


def _call_api(
    client, model: str, img_bytes: bytes, media_type: str,
    user_text: str, system: str, max_tokens: int, temperature: float = 0.0,
) -> str | None:
    """Make one API call with up to 5 retries on transient errors. Returns raw text or 'api_error:...'."""
    import random
    delays = [2, 8, 30, 90]  # seconds between attempts 1→2, 2→3, 3→4, 4→5
    for attempt in range(5):
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
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                resp_text = resp.text
                return resp_text.strip() if resp_text else None
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
            if any(t in msg_str for t in _TRANSIENT) and attempt < 4:
                delay = delays[attempt] * (0.8 + 0.4 * random.random())  # ±20% jitter
                click.echo(f"  [API] transient error (attempt {attempt+1}/5), retrying in {delay:.0f}s…")
                time.sleep(delay)
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
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            # Only accept JSON objects; bare null / string / array → treat as parse failure
            return None
        # Normalize string "null" → Python None (VLM sometimes returns "null" as a string)
        if isinstance(parsed.get("result"), str) and parsed["result"].strip().lower() == "null":
            parsed["result"] = None
        return parsed
    except json.JSONDecodeError:
        # Salvage path 1: totals cell — extract R/H/E/LOB from truncated JSON
        totals_keys = ("R", "H", "E", "LOB")
        totals_found = {}
        for k in totals_keys:
            m = re.search(rf'"{k}"\s*:\s*(-?\d+)', raw)
            if m:
                totals_found[k] = int(m.group(1))
        if totals_found:
            return {k: totals_found.get(k, 0) for k in totals_keys}

        # Salvage path 2: PA cell — extract result/run/notes
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
    col_to_inning: list[int] | None = None,
) -> dict[int, int]:
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

    # ── Rule A: hole in lineup — flag any remaining holes after the pre-read pass ─
    # _reread_hole_cells runs before batting rules and fills in most holes via VLM.
    # Any still-null holes here get flagged (VLM couldn't read them either).
    for ci in range(innings):
        inn = col_to_inning[ci] if col_to_inning else ci + 1
        for ri in range(n_rows):
            if _has_pa(ri, ci) or _is_uncertain(ri, ci):
                continue
            prev_ri = (ri - 1) % n_rows
            next_ri = (ri + 1) % n_rows
            if _has_pa(prev_ri, ci) and _has_pa(next_ri, ci):
                cell = grid[ri][ci] or {}
                run_flag = cell.get("run", False)
                click.echo(
                    f"  [hole] P{ri+1} inn {inn}: still null between P{prev_ri+1} "
                    f"and P{next_ri+1} after re-read"
                    + (" (run=True!)" if run_flag else "")
                )

    # Derive overflow column set from col_to_inning (same inning as preceding col).
    _overflow_ci: set[int] = set()
    if col_to_inning:
        for _ci in range(1, innings):
            if col_to_inning[_ci] == col_to_inning[_ci - 1]:
                _overflow_ci.add(_ci)

    # ── Rule 1: outs can never score a run ────────────────────────────────────
    for ri in range(n_rows):
        for ci in range(innings):
            inn = col_to_inning[ci] if col_to_inning else ci + 1
            cell = grid[ri][ci]
            if cell and cell.get("run") and _is_out(cell.get("result")):
                cell["run"] = False
                _save(ri, ci)
                click.echo(f"  [out-run] Player {ri+1} inn {inn} {cell['result']}: run forced False")

    # ── Rule 2: remove isolated PAs ───────────────────────────────────────────
    # Skip overflow columns entirely — they legitimately have few batters.
    # Skip if either neighbor is uncertain (api/parse error) — we can't confirm it's truly isolated.
    for ci in range(innings):
        if ci in _overflow_ci:
            continue  # overflow column: isolated check doesn't apply
        inn = col_to_inning[ci] if col_to_inning else ci + 1
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
                click.echo(f"  [isolated] Player {ri+1} inn {inn} was {old}: removed")

    # ── Rule 3: three-out rule ────────────────────────────────────────────────
    # Inning 1 starts at player 1 (row 0).
    # Each subsequent inning starts at (last batter of previous inning + 1) % n_rows.
    # Only applied when all cells in the inning are confirmed (no api/parse errors),
    # since uncertain cells break start-player tracking.
    # When a column wraps (overflow), outs carry over from the previous column.
    start_ri = 0
    outs_by_inning: dict[int, int] = {}  # tracks cumulative outs per inning across overflow cols
    last_batter_by_inning: dict[int, int] = {}  # inning → 1-based batting slot of last batter
    for ci in range(innings):
        inn = col_to_inning[ci] if col_to_inning else ci + 1
        # Skip this column if any cell is uncertain
        if any(_is_uncertain(ri, ci) for ri in range(n_rows)):
            for k in range(n_rows):
                ri = (start_ri + k) % n_rows
                if _has_pa(ri, ci):
                    start_ri = (ri + 1) % n_rows
            continue
        outs = outs_by_inning.get(inn, 0)  # carry from previous col if same inning (overflow)
        last_batter_ri: int | None = None
        for k in range(n_rows):
            ri = (start_ri + k) % n_rows
            if not _has_pa(ri, ci):
                continue
            if outs >= 3:
                old = grid[ri][ci].get("result")
                grid[ri][ci] = {"result": None, "run": False, "notes": f"removed:after_3_outs ({old})"}
                _save(ri, ci)
                click.echo(f"  [3-outs] Player {ri+1} inn {inn} col {ci+1} was {old}: removed")
            else:
                last_batter_ri = ri
                if _is_out(grid[ri][ci].get("result")):
                    outs += 1
        outs_by_inning[inn] = outs
        if last_batter_ri is not None:
            last_batter_by_inning[inn] = last_batter_ri + 1  # 1-based slot
            start_ri = (last_batter_ri + 1) % n_rows
    return last_batter_by_inning


def _reread_run_no_result_cells(
    img: np.ndarray,
    grid: list[list[dict | None]],
    n_rows: int,
    n_cols: int,
    row_tops: list[int],
    row_bottoms: list[int],
    col_lefts: list[int],
    col_to_inning: list[int],
    row_names: list[dict],
    client,
    model: str,
    cache_dir: Path,
) -> int:
    """Re-read any cell where run=True but result=None.

    The VLM detected a run marker but missed the PA result — do a fresh
    classify_cell call (bypassing cache) to fill in the result.
    Returns the number of cells re-read.
    """
    reread = 0
    for ri in range(n_rows):
        for ci in range(n_cols):
            cell = grid[ri][ci]
            if not cell or cell.get("result") is not None:
                continue
            if not cell.get("run"):
                continue
            inn = col_to_inning[ci]
            names = row_names[ri].get("players") or [] if ri < len(row_names) else []
            player_name = names[0] if names else f"P{ri+1}"
            y1, y2 = max(0, row_tops[ri]), row_bottoms[ri]
            x1 = max(0, col_lefts[ci])
            x2 = min(img.shape[1], col_lefts[ci + 1] if ci + 1 < len(col_lefts) else img.shape[1])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            click.echo(
                f"  [run-reread] P{ri+1} inn {inn}: run=True but result=None — re-reading cell…"
            )
            # Use scale=8 and a focused prompt: the run marker is already confirmed,
            # so the model only needs to find the PA result in BOTTOM-RIGHT.
            img_bytes, media_type = _encode_cell(crop, scale=8)
            user_text = (
                f"Player: {player_name}  Inning: {inn}\n"
                "A run-scored marker IS present in this cell (already confirmed). "
                "Focus on the BOTTOM-RIGHT quadrant to find the plate appearance result. "
                "There MUST be a result here — a hit stroke, out circle, walk humps, or error. "
                "Return JSON only."
            )
            raw = _call_api(client, model, img_bytes, media_type, user_text,
                            _CELL_SYSTEM, max_tokens=400, temperature=0.0)
            result = None
            if raw and not raw.startswith("api_error:"):
                result = _parse_json_response(raw)
            if result and result.get("result") is not None:
                result["run"] = True  # preserve the confirmed run
                grid[ri][ci] = result
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                click.echo(f"    → result: {result['result']}  run: {result.get('run')}")
                reread += 1
            else:
                click.echo(f"    → still no result after re-read ({(raw or 'None')[:80]})")
    return reread


def _reread_hole_cells(
    img: np.ndarray,
    grid: list[list[dict | None]],
    n_rows: int,
    n_cols: int,
    row_tops: list[int],
    row_bottoms: list[int],
    col_lefts: list[int],
    col_to_inning: list[int],
    row_names: list[dict],
    client,
    model: str,
    cache_dir: Path,
) -> int:
    """Re-read null cells that are sandwiched between two non-null cells in the
    same column (batting order is continuous — skipping a batter is impossible).
    Returns the number of cells re-read.
    """
    def _has_result(ri: int, ci: int) -> bool:
        c = grid[ri][ci]
        return c is not None and c.get("result") is not None

    reread = 0
    for ci in range(n_cols):
        inn = col_to_inning[ci] if col_to_inning else ci + 1
        for ri in range(n_rows):
            if _has_result(ri, ci):
                continue
            prev_ri = (ri - 1) % n_rows
            next_ri = (ri + 1) % n_rows
            if not (_has_result(prev_ri, ci) and _has_result(next_ri, ci)):
                continue
            names = row_names[ri].get("players") or [] if ri < len(row_names) else []
            player_name = names[0] if names else f"P{ri+1}"
            y1, y2 = max(0, row_tops[ri]), row_bottoms[ri]
            x1 = max(0, col_lefts[ci])
            x2 = min(img.shape[1], col_lefts[ci + 1] if ci + 1 < len(col_lefts) else img.shape[1])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            click.echo(
                f"  [hole-reread] P{ri+1} ({player_name}) inn {inn}: "
                f"null between P{prev_ri+1} and P{next_ri+1} — re-reading…"
            )
            img_bytes, media_type = _encode_cell(crop, scale=8)
            user_text = (
                f"Player: {player_name}  Inning: {inn}\n"
                "The batters immediately before and after this player both have "
                "plate appearances in this inning, so this player MUST have batted too. "
                "Look very carefully — the mark may be faint or small. "
                "Return JSON only."
            )
            raw = _call_api(client, model, img_bytes, media_type, user_text,
                            _CELL_SYSTEM, max_tokens=400, temperature=0.0)
            result = None
            if raw and not raw.startswith("api_error:"):
                result = _parse_json_response(raw)
            cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
            if result and result.get("result") is not None:
                grid[ri][ci] = result
                cf.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                click.echo(f"    → result: {result['result']}  run: {result.get('run', False)}")
                reread += 1
            else:
                click.echo(f"    → still no result after re-read ({(raw or 'None')[:80]})")
    return reread


def _enforce_pa_ordering(lineup: list) -> list[str]:
    """PA counts must be non-increasing by batting order.

    The lead-off slot gets the most plate appearances; each later slot the same
    or fewer. When a later slot exceeds the previous slot's count, remove the
    excess PAs — lowest-confidence first, then highest inning (most likely a
    phantom from ink bleed or an off-by-one column assignment).
    """
    conf_rank = {"low": 0, "high": 2}
    msgs: list[str] = []
    prev_cap: int | None = None

    for slot in sorted(lineup, key=lambda s: s.batting_order):
        # Collect (pa, player) pairs across all players in this batting slot
        pa_player: list[tuple] = [
            (pa, player)
            for player in slot.players
            for pa in player.plate_appearances
        ]
        count = len(pa_player)

        if prev_cap is not None and count > prev_cap:
            excess = count - prev_cap
            ranked = sorted(
                pa_player,
                key=lambda x: (conf_rank.get(x[0].confidence, 2), -x[0].inning),
            )
            remove_ids = {id(pa) for pa, _ in ranked[:excess]}
            removed_strs = []
            for player in slot.players:
                kept = []
                for pa in player.plate_appearances:
                    if id(pa) in remove_ids:
                        removed_strs.append(f"inn {pa.inning} {pa.result}")
                    else:
                        kept.append(pa)
                player.plate_appearances = kept
                player.summary = _make_summary(player.plate_appearances)
            msgs.append(
                f"  [pa-order] slot {slot.batting_order}: trimmed {excess} PA "
                f"({count}→{prev_cap}): " + ", ".join(removed_strs)
            )
            count = prev_cap

        prev_cap = count

    return msgs


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
    line = f"  [Slot {slot:2d}] {name:<22}  PA={pa:>2}  AB={ab:>2}  H={h:>2}  R={r:>2}"
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
    line = f"  [Inn {inning:<2}]  PA={pa:>3}  R={r:>2}  H={h:>2}  E={e:>2}  Outs={outs}"
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
    col_to_inning: list[int] | None = None,
) -> None:
    """Post-process: use GT run totals to enforce impossible runs.
    - GT R=0 for an inning → force all run=True cells to False.
    - GT R < extracted → log a warning; cannot auto-reduce without per-cell GT.
    Aggregates across overflow columns (multiple cols can map to the same inning).
    """
    from collections import defaultdict
    inning_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ci in range(innings):
        inn = col_to_inning[ci] if col_to_inning else ci + 1
        for ri in range(n_rows):
            inning_cells[inn].append((ri, ci))

    for inn, cells in sorted(inning_cells.items()):
        if inn not in gt_totals:
            continue
        gt_r = gt_totals[inn]["R"]
        extracted = [(ri, ci) for ri, ci in cells if (grid[ri][ci] or {}).get("run")]
        if gt_r == 0 and extracted:
            for ri, ci in extracted:
                grid[ri][ci]["run"] = False
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")
                click.echo(f"  [GT-R=0] Player {ri+1} inn {inn}: run forced False (GT=0 runs)")
        elif len(extracted) > gt_r:
            click.echo(
                f"  [GT-R] Inn {inn}: {len(extracted)} runs extracted, GT={gt_r} — cannot auto-reduce"
            )


# ── Player name detection ─────────────────────────────────────────────────────

_NAME_SYSTEM = """\
You are reading the player information strip on the left side of a Dutch KNBSB baseball scorecard row.
The strip is divided into up to THREE horizontal sub-rows (top to bottom):
  1. Starter — always present
  2. First substitute — present if a sub entered during the game
  3. Second substitute — present if a second sub entered later
Read ALL sub-rows from top to bottom and report each name you see.
IMPORTANT: a block of small printed numbers (like '3 1 2') below a name is pitcher statistics \
— NOT a player name. Jersey numbers like '15' or '2/5' next to a name are NOT separate players.
If you see only 1 or 2 names, leave the remaining positions null.
Return ONLY valid JSON (no prose, no markdown):
{"players": ["<starter name>", "<sub1 name or null>", "<sub2 name or null>"]}
Always include exactly 3 elements. Use null for missing positions."""

_SINGLE_NAME_SYSTEM = """\
You are reading ONE sub-row from the player name strip of a Dutch KNBSB baseball scorecard.
This sub-row may contain a handwritten player name (sometimes with a jersey number beside it).
IMPORTANT: a block of small printed numbers (like '3 1 2') is pitcher statistics — NOT a name.
Jersey numbers (like '15' or '2/5') beside a name are NOT separate players.
The user message may include a list of known players on this team.
If the handwriting is partial or hard to read, match what you can see against that list and return the full name of the best match.
If no name is visible and nothing in the list matches, return the word null.
Return ONLY the player name — no explanation, no JSON."""

_SUB_INNING_SYSTEM = """\
You are looking at the batting-grid row for a single player on a Dutch KNBSB baseball scorecard.
The row spans inning columns 1 through N.
When a player was substituted, the scorekeeper drew a squiggly (wavy) vertical line at the LEFT \
edge of the inning in which the substitute entered.
Return ONLY valid JSON: {"sub_inning": <1-based column number where the squiggly line appears, or null>}
If you see no squiggly line, return: {"sub_inning": null}"""

_TOTALS_ROW_SYSTEM = """\
You are reading a single per-inning totals cell from a Dutch KNBSB baseball scorecard.
The cell has a 2×2 quadrant layout:
  TOP-LEFT      = E   (errors this inning)
  TOP-RIGHT     = H   (hits this inning)
  BOTTOM-LEFT   = LOB (runners left on base)
  BOTTOM-RIGHT  = R   (runs scored this inning)
Each quadrant contains a single handwritten integer. A blank quadrant means 0.
Return ONLY valid JSON — no prose, no markdown:
{"R": <int>, "H": <int>, "E": <int>, "LOB": <int>}"""


def _detect_inning_totals(
    img: np.ndarray,
    extra_tops: list[int],
    extra_bottoms: list[int],
    col_lefts: list[int],
    n_phys_cols: int,
    col_to_inning: list[int],
    client,
    model: str,
    cache_dir: Path,
) -> dict[int, dict]:
    """VLM-detect per-inning totals from the first extra row below the batting grid.

    Strategy: send the ENTIRE totals row as one image (1 API call).  This avoids
    the parallel-call rate-limit bursts that occur right after PA cell extraction.
    Falls back to per-cell sequential calls if the full-row parse is incomplete.

    Returns {inning: {"R": int, "H": int, "E": int, "LOB": int}}.
    Cached in cells/_totals_raw.json.
    """
    if not extra_tops or not extra_bottoms:
        return {}

    cache_path = cache_dir / "_totals_raw.json"
    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            # Drop entries that are missing required keys (from a previous truncated run)
            for k, v in raw_cache.items():
                if isinstance(v, dict) and "R" in v and "H" in v:
                    cached[k] = v
        except (json.JSONDecodeError, OSError):
            cached = {}

    img_h = img.shape[0]
    y1 = max(0, extra_tops[0])
    y2 = min(img_h, extra_bottoms[0])
    x1 = max(0, col_lefts[0])
    x2 = min(img.shape[1], col_lefts[n_phys_cols] if n_phys_cols < len(col_lefts) else img.shape[1])

    if y2 <= y1:
        click.echo(f"  Totals row out of image bounds (y={y1}–{y2}, img_h={img_h}) — skipping.")
        return {}

    missing = [ci for ci in range(n_phys_cols) if f"c{ci+1:02d}" not in cached]
    if missing:
        # Brief settle delay — PA-cell extraction fires ~90 parallel calls just before
        # this; a short pause lets the API rate-limit window reset.
        click.echo(f"  Waiting 4 s for API to settle after PA extraction…")
        time.sleep(4)

        # Sequential per-cell calls — one crop per inning, no parallelism.
        click.echo(f"  Reading {len(missing)} totals cell(s) sequentially…")
        for ci in missing:
            cx1 = max(0, col_lefts[ci])
            cx2 = min(img.shape[1], col_lefts[ci + 1] if ci + 1 < len(col_lefts) else img.shape[1])
            crop = img[y1:y2, cx1:cx2]
            if crop.size == 0:
                click.echo(f"  Inn {ci+1}: empty crop, skipping.")
                continue
            img_bytes, media_type = _encode_cell(crop, scale=4)
            raw = _call_api(
                client, model, img_bytes, media_type,
                "Read the four quadrant values. Return JSON only.",
                _TOTALS_ROW_SYSTEM, max_tokens=512, temperature=0.0,
            )
            if raw and not raw.startswith("api_error:"):
                parsed = _parse_json_response(raw)
                if parsed and "R" in parsed and "H" in parsed:
                    key = f"c{ci+1:02d}"
                    cached[key] = {k: max(0, int(parsed.get(k) or 0)) for k in ("R", "H", "E", "LOB")}
                    inn_label = col_to_inning[ci]
                    click.echo(f"  Col {ci+1} (Inn {inn_label}): R={cached[key]['R']} H={cached[key]['H']} E={cached[key]['E']} LOB={cached[key]['LOB']}")
                else:
                    click.echo(f"  Col {ci+1}: unexpected response: {(raw or '')[:80]}")
            else:
                click.echo(f"  Col {ci+1}: API error: {(raw or 'None')[:120]}")

        cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")

    # Map columns → innings.
    # For wrapped innings the scribe writes the full inning totals in the LAST
    # (overflow) column; earlier columns for the same inning are blank/zero.
    # R is accumulated across all columns (runs may be noted per-column);
    # H/E/LOB are updated with the last non-zero value seen for the inning.
    out: dict[int, dict] = {}
    for ci in range(n_phys_cols):
        key = f"c{ci+1:02d}"
        if key not in cached:
            continue
        inn = col_to_inning[ci]
        vals = cached[key]
        if inn not in out:
            out[inn] = dict(vals)
        else:
            out[inn]["R"] += vals["R"]
            if vals["H"]   > 0: out[inn]["H"]   = vals["H"]
            if vals["E"]   > 0: out[inn]["E"]   = vals["E"]
            if vals["LOB"] > 0: out[inn]["LOB"] = vals["LOB"]
    return out


def _interactive_review_totals(totals: dict[int, dict], max_inn: int) -> dict[int, dict]:
    """Print the detected totals table and let the user correct any row."""
    click.echo()
    click.echo("  Inn    R    H    E   LOB")
    click.echo("  " + "-" * 28)
    for inn in range(1, max_inn + 1):
        t = totals.get(inn, {})
        click.echo(
            f"  {inn:3d}  {t.get('R','?')!s:>4} {t.get('H','?')!s:>4}"
            f" {t.get('E','?')!s:>4} {t.get('LOB','?')!s:>5}"
        )
    click.echo()
    click.echo("  To correct a row enter: inning R H E LOB  (e.g. '3 2 4 1 0')")
    click.echo("  Press Enter to continue.")
    while True:
        raw = click.prompt("  Edit", default="").strip()
        if not raw:
            break
        parts = raw.split()
        if len(parts) == 5 and all(p.lstrip("-").isdigit() for p in parts):
            inn, r, h, e, lob = (int(p) for p in parts)
            totals[inn] = {"R": r, "H": h, "E": e, "LOB": lob}
            click.echo(f"    Updated inning {inn}: R={r} H={h} E={e} LOB={lob}")
        else:
            click.echo("  Expected 5 integers: inning R H E LOB")
    return totals


def _write_totals_txt(game_dir: Path, stem: str, totals: dict[int, dict], innings: int) -> Path:
    """Write _totals.txt from auto-detected totals data."""
    path = game_dir / f"{stem}_totals.txt"
    lines = [
        "# Totaal per inning — auto-extracted from scan; edit if needed, then re-run.",
        "# inning  runs  hits  errors  lob",
    ]
    for inn in range(1, innings + 1):
        t = totals.get(inn)
        if t:
            lines.append(f"{inn}\t{t['R']}\t{t['H']}\t{t['E']}\t{t['LOB']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    click.echo(f"  Written: {path.name}")
    return path


def _detect_row_names(
    img: np.ndarray,
    row_tops: list[int],
    row_bottoms: list[int],
    col_lefts: list[int],
    n_rows: int,
    client,
    model: str,
    cache_dir: Path,
    roster: list[tuple[str, int | None]] | None = None,
) -> list[dict]:
    """
    VLM-based player name detection from the left info strip of each row.
    Returns list of {"players": [name, ...]}, 1–3 names per row.
    Cached in cache_dir/_names.json; re-uses existing entries.
    """
    cache_path = cache_dir / "_names.json"
    cached: dict = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    x_right = col_lefts[0] if col_lefts else img.shape[1]
    x_left = int(img.shape[1] * 0.03)  # skip ring-binder margin

    roster_hint = ""
    if roster:
        roster_hint = "  Known players: " + ", ".join(n for n, _ in roster) + "."

    results: list[dict] = []

    for ri in range(n_rows):
        key = f"r{ri + 1:02d}"
        if key in cached:
            raw_cached = cached[key]
            # Normalise legacy {starter, sub} entries on the fly
            if "starter" in raw_cached:
                ps = [p for p in [raw_cached.get("starter"), raw_cached.get("sub")] if p]
                raw_cached = {"players": ps}
            if raw_cached.get("players"):
                results.append(raw_cached)
                continue
            # Null/empty cached entry — retry VLM

        y1 = max(0, row_tops[ri])
        y2 = row_bottoms[ri]
        full_strip = img[y1:y2, x_left:min(x_right, img.shape[1])]

        if full_strip.size == 0:
            results.append({"players": []})
            continue

        # Primary: full strip at scale=2 → 288×1888px for a 944×144 source.
        # No width clip — avoids truncating longer names in future games.
        img_bytes, media_type = _encode_cell(full_strip, scale=2)
        raw = _call_api(
            client, model, img_bytes, media_type,
            f"Read ALL player names in this strip (top to bottom). Return JSON only.{roster_hint}",
            _NAME_SYSTEM, max_tokens=300, temperature=1.0,
        )
        if raw and not raw.startswith("api_error:"):
            parsed = _parse_json_response(raw)
            if parsed and "players" in parsed:
                plist = parsed["players"]
                entry: dict = {"players": [p for p in (plist or []) if p]}
            elif parsed and "starter" in parsed:
                ps = [p for p in [parsed.get("starter"), parsed.get("sub")] if p]
                entry = {"players": ps}
            else:
                entry = None
        else:
            entry = None

        # Fallback: if the full-strip call returned nothing, try each of the
        # three sub-rows independently.  Clip each to 700px wide (covers any
        # realistic handwritten name) then scale height to 200px → ~200×700px
        # per sub-row — a 3.5:1 aspect ratio the VLM handles well.
        if not entry or not entry.get("players"):
            h_strip = full_strip.shape[0]
            sub_h = max(1, h_strip // 3)
            sub_names: list[str] = []
            for si in range(3):
                y_s = si * sub_h
                y_e = min(h_strip, (si + 1) * sub_h)
                sub_crop = full_strip[y_s:y_e, : min(700, full_strip.shape[1])]
                tw = sub_crop.shape[1]
                sub_big = cv2.resize(sub_crop, (tw, 200), interpolation=cv2.INTER_CUBIC)
                ok2, buf2 = cv2.imencode(".jpg", sub_big, [cv2.IMWRITE_JPEG_QUALITY, 92])
                hint_text = f" {roster_hint.strip()}" if roster_hint.strip() else ""
                s_raw = _call_api(
                    client, model, buf2.tobytes(), "image/jpeg",
                    f"What is the player name written in this row?{hint_text}",
                    _SINGLE_NAME_SYSTEM, max_tokens=60, temperature=1.0,
                )
                if s_raw and not s_raw.startswith("api_error:") and s_raw.strip().lower() != "null":
                    sub_names.append(s_raw.strip())
            if sub_names:
                entry = {"players": sub_names}

        if not entry or not entry.get("players"):
            # No usable result — don't cache, retry on next run
            results.append({"players": []})
            continue
        results.append(entry)
        cached[key] = entry

    cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _interactive_review_names(row_names: list[dict], cache_dir: Path) -> list[dict]:
    """
    Show a table of all detected player names and let the user correct any row
    before name resolution proceeds.  Corrections are written back to the cache.
    """
    def _display(rn: list[dict]) -> None:
        click.echo("\n  Slot  Names detected (starter  |  sub1  |  sub2)")
        click.echo("  " + "-" * 60)
        for ri2, e in enumerate(rn):
            ps = list(e.get("players") or [])
            row_str = "  |  ".join(ps) if ps else "(not detected)"
            click.echo(f"  {ri2 + 1:>4}.  {row_str}")
        click.echo()

    _display(row_names)
    click.echo("  Enter a slot number to edit it, or press Enter to continue.")

    while True:
        raw = click.prompt("  Edit slot", default="").strip()
        if not raw:
            break
        if not raw.isdigit() or not (1 <= int(raw) <= len(row_names)):
            click.echo(f"  Please enter a number between 1 and {len(row_names)}.")
            continue
        ri = int(raw) - 1
        current = list(row_names[ri].get("players") or [])
        new_players: list[str] = []
        labels = ["Starter", "Sub 1 ", "Sub 2 "]
        for pi in range(3):
            default_val = current[pi] if pi < len(current) else ""
            val = click.prompt(
                f"    {labels[pi]} [{default_val or 'none'}]",
                default=default_val,
            ).strip()
            if val:
                new_players.append(val)
            else:
                break  # no more players in this slot

        row_names[ri] = {"players": new_players}

        # Persist correction to cache so --reuse-cache picks it up next time
        cache_path = cache_dir / "_names.json"
        cached: dict = {}
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        key = f"r{ri + 1:02d}"
        if new_players:
            cached[key] = {"players": new_players}
        elif key in cached:
            del cached[key]
        cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"  Slot {ri + 1} updated: {new_players}")
        _display(row_names)

    return row_names


def _fuzzy_match_name(
    detected: str | None,
    roster: list[tuple[str, int | None]],
    threshold: int = 60,
) -> tuple[str, int | None] | None:
    """Fuzzy-match a VLM-detected name against the full roster. Returns matched entry or None."""
    if not detected:
        return None
    from rapidfuzz import process, fuzz
    names = [n for n, _ in roster]
    result = process.extractOne(detected, names, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        idx = names.index(result[0])
        return roster[idx]
    return None


def _prompt_player_selection(
    detected: str | None,
    roster: list[tuple[str, int | None]],
    row_label: str,
    context: str = "starter",
) -> tuple[str, int | None]:
    """
    Show the full roster and ask the user to pick the correct player (or add a new one).
    Updates players.txt if a new player is entered.
    """
    click.echo()
    if detected:
        click.echo(f"  Row {row_label} {context}: VLM read '{detected}' but it didn't match any roster player.")
    else:
        click.echo(f"  Row {row_label} {context}: VLM could not read the name.")
    click.echo()
    click.echo("  Select player:")
    for i, (pname, jersey) in enumerate(roster, 1):
        tag = f"  #{jersey}" if jersey else ""
        click.echo(f"    {i:>2}.  {pname}{tag}")
    new_idx = len(roster) + 1
    click.echo(f"    {new_idx:>2}.  Enter new player name")
    click.echo()

    while True:
        raw = click.prompt("  Choice", default="", show_default=False, prompt_suffix=" ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(roster):
                chosen = roster[idx - 1]
                click.echo(f"  Assigned: {chosen[0]}")
                return chosen
            if idx == new_idx:
                break
        click.echo(f"  Enter a number between 1 and {new_idx}.")

    # New player
    new_name = click.prompt("  New player name").strip()
    jersey_raw = click.prompt("  Jersey number (leave blank to skip)", default="").strip()
    jersey_int = int(jersey_raw) if jersey_raw.isdigit() else None

    from db import _get_data_root
    players_txt = _get_data_root() / "players.txt"
    jersey_suffix = f", {jersey_int}" if jersey_int is not None else ""
    with open(players_txt, "a", encoding="utf-8") as f:
        f.write(f"{new_name}{jersey_suffix}\n")
    click.echo(f"  Created '{new_name}' and added to {players_txt.name}")

    roster.append((new_name, jersey_int))
    return (new_name, jersey_int)


def _update_names_cache(cache_dir: Path, ri: int, player_index: int, name: str) -> None:
    """Persist a manual name correction into the names cache so re-runs skip the prompt."""
    cache_path = cache_dir / "_names.json"
    cached: dict = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    key = f"r{ri + 1:02d}"
    entry = cached.get(key, {})
    if "starter" in entry:
        ps = [p for p in [entry.get("starter"), entry.get("sub")] if p]
        entry = {"players": ps}
    players = list(entry.get("players") or [])
    while len(players) <= player_index:
        players.append(None)
    players[player_index] = name
    cached[key] = {"players": [p for p in players if p]}
    cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")


def _detect_sub_inning_vlm(
    img: np.ndarray,
    ri: int,
    row_tops: list[int],
    row_bottoms: list[int],
    col_lefts: list[int],
    innings: int,
    client,
    model: str,
) -> int | None:
    """
    Ask the VLM to identify the inning column where a squiggly substitution line appears.
    Returns 1-based inning number, or None if not detected.
    """
    y1 = max(0, row_tops[ri])
    y2 = row_bottoms[ri]
    x1 = col_lefts[0] if col_lefts else 0
    x2 = col_lefts[innings] if innings < len(col_lefts) else img.shape[1]
    row_strip = img[y1:y2, x1:min(x2, img.shape[1])]
    if row_strip.size == 0:
        return None
    img_bytes, media_type = _encode_cell(row_strip, scale=3)
    raw = _call_api(
        client, model, img_bytes, media_type,
        f"This row has {innings} inning columns. Find the squiggly sub line. Return JSON only.",
        _SUB_INNING_SYSTEM, max_tokens=30,
    )
    if raw and not raw.startswith("api_error:"):
        parsed = _parse_json_response(raw)
        if parsed:
            inn = parsed.get("sub_inning")
            if isinstance(inn, int) and 1 <= inn <= innings:
                return inn
    return None


def _make_summary(pas: list) -> "PASummary":
    h  = sum(1 for pa in pas if _is_hit(pa.result))
    ab = sum(1 for pa in pas if _is_ab(pa.result))
    r  = sum(1 for pa in pas if pa.run_scored)
    return PASummary(PA=len(pas), AB=ab, H=h, R=r)


# ── Log tee ───────────────────────────────────────────────────────────────────

class _TeeWriter:
    """Write to two streams simultaneously so console output also lands in a log file."""
    def __init__(self, primary, secondary):
        self._p, self._s = primary, secondary

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode(getattr(self._p, "encoding", "utf-8"), errors="replace")
        self._p.write(data)
        self._s.write(data)
        return len(data)

    def flush(self):
        self._p.flush()
        self._s.flush()

    def fileno(self):
        return self._p.fileno()

    def isatty(self):
        return False

    def __getattr__(self, name):
        # Don't expose .buffer — Click 8.4+ writes bytes directly to .buffer,
        # which would bypass this tee. Hiding it forces Click to use write().
        if name == "buffer":
            raise AttributeError(name)
        return getattr(self._p, name)


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
@click.option("--model", default=None,
              help="Anthropic model (default: EXTRACTION_MODEL env or claude-sonnet-4-6).")
@click.option("--workers", default=8, show_default=True, help="Parallel VLM calls.")
@click.option("--reuse-cache", is_flag=True, default=False,
              help="Reuse cached per-cell results (skip API calls for cached cells).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Classify cells and check integrity but do not write to DB.")
@click.option("--yes", "-y", "auto_yes", is_flag=True, default=False,
              help="Skip all interactive review prompts; use cached / auto values throughout.")
@click.option("--gt-dir", default=None,
              help="Ground-truth directory (default: the game folder inside data_root/games/).")
def main(
    image_path, players_file, active_players, innings,
    n_player_rows, model, workers,
    reuse_cache, dry_run, auto_yes, gt_dir,
):
    """Cell-based scorecard extraction — inning from column position, not VLM guessing."""
    img_path = Path(image_path).resolve()
    data_root = img_path.parent.parent   # Quick 2026 data/ (image lives in data_root/scans/)
    game_dir = data_root / "games" / img_path.stem
    game_dir.mkdir(parents=True, exist_ok=True)

    # Tee console output to a log file in the game folder.
    import atexit
    _log_path = game_dir / f"{img_path.stem}_run.log"
    _log_f = open(_log_path, "w", encoding="utf-8", errors="replace")
    _orig_stdout = sys.stdout
    sys.stdout = _TeeWriter(_orig_stdout, _log_f)

    def _close_log():
        sys.stdout = _orig_stdout   # restore before Python's own shutdown flush
        try:
            _log_f.flush()
            _log_f.close()
        except OSError:
            pass

    atexit.register(_close_log)

    if model is None:
        model = os.environ.get("EXTRACTION_MODEL", "claude-sonnet-4-6")
    click.echo(f"Image : {img_path.name}")
    click.echo(f"Model : {model}   Workers: {workers}   Innings: {innings}")

    # ── Date / opponent from filename ─────────────────────────────────────────
    # Handles both formats:
    #   YYYY-MM-DD_opponent          (old: 2026-06-07_almere)
    #   YYYY-MM-DD - Opponent (Side) (new: 2026-04-12 - Thamen (Home))
    date_str, opponent = None, None
    date_m = re.match(r"(\d{4}-\d{2}-\d{2})", img_path.stem)
    if date_m:
        date_str = date_m.group(1)
        rest = img_path.stem[len(date_str):].strip()
        rest = re.sub(r"^[\s_\-–]+", "", rest).strip()          # strip leading separators
        rest = re.sub(r"\s*\((Home|Away)\)\s*$", "", rest, flags=re.IGNORECASE).strip()
        if rest:
            opponent = rest.replace("_", " ")
    if date_str:
        click.echo(f"Date  : {date_str}   Opponent: {opponent or 'Unknown'}")

    # ── Roster (batting-order slots 1..active_players) ────────────────────────
    # One line per batting slot in order; subs share a slot and are noted
    # in supplementary data — they do NOT get their own grid row.
    # Per-game roster auto-discovery: look for {stem}.txt in the scan dir or a
    # rosters/ subfolder, falling back to the --players option.
    roster: list[tuple[str, int | None]] = []  # (name, jersey)
    _per_game_candidates = [
        img_path.parent / f"{img_path.stem}.txt",
        img_path.parent / "rosters" / f"{img_path.stem}.txt",
    ]
    # Resolve --players: check per-game candidates first, then data_root, then CWD/repo root.
    _players_abs = Path(players_file)
    if not _players_abs.exists():
        _players_abs = data_root / players_file
    if not _players_abs.exists():
        _players_abs = data_root / "players.txt"
    players_path = next((p for p in _per_game_candidates if p.exists()), _players_abs)
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
    click.echo(f"Roster: [{players_path.name}] {', '.join(n for n, _ in active_roster)}")

    # ── Grid detection ────────────────────────────────────────────────────────
    click.echo("\nDetecting grid...")
    debug_img = str(game_dir / f"{img_path.stem}_grid_debug.png")
    row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts, cell_size = detect_grid(
        str(img_path),
        n_player_rows=n_player_rows,
        debug_out=debug_img,
    )
    img = cv2.imread(str(img_path))
    n_active_rows = min(active_players, len(row_tops))
    # Physical columns to read: all detected scoring columns, capped at
    # game innings + 2 (enough buffer for any wrap columns).
    n_phys_cols = min(len(col_lefts) - 1, innings + 2)
    click.echo(f"Grid  : {len(row_tops)} player rows x {len(col_lefts)-1} cols, cell={cell_size}px")
    click.echo(f"Active: first {n_active_rows} rows ({innings} game innings, {n_phys_cols} cols to read)")
    click.echo(f"Debug : {debug_img}")

    # col_to_inning is built after VLM classification; default to 1:1 for now
    col_to_inning: list[int] = list(range(1, n_phys_cols + 1))

    # ── Cache directory ───────────────────────────────────────────────────────
    cache_dir = game_dir / "cells"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── API client (needed for both name detection and cell classification) ───
    if model.startswith("gemini"):
        from google import genai as google_genai
        client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Player name detection from scan ──────────────────────────────────────
    click.echo("\nDetecting player names from scan...")
    row_names = _detect_row_names(
        img, row_tops, row_bottoms, col_lefts, n_active_rows, client, model, cache_dir,
        roster=roster,
    )
    if not auto_yes:
        row_names = _interactive_review_names(row_names, cache_dir)

    # Build slot_info: for each row, resolved starter + subs with entry innings.
    # subs = [(player_tuple, entry_inning), ...]  (empty list if no substitutions)
    slot_info: list[dict] = []
    for ri, names in enumerate(row_names):
        detected_players: list[str] = names.get("players") or []
        # Normalise legacy {starter, sub} format
        if not detected_players:
            if names.get("starter"):
                detected_players = [names["starter"]]
                if names.get("sub"):
                    detected_players.append(names["sub"])

        resolved: list[tuple[str, int | None]] = []
        for pi, detected in enumerate(detected_players):
            matched = _fuzzy_match_name(detected, roster)
            if matched:
                resolved.append(matched)
                label = "starter" if pi == 0 else f"sub{pi}"
                click.echo(f"  Row {ri+1} {label}: '{detected}' → {matched[0]} (#{matched[1]})")
            else:
                context = "sub" if pi > 0 else None
                chosen = _prompt_player_selection(detected, roster, str(ri + 1), context=context)
                _update_names_cache(cache_dir, ri, pi, chosen[0])
                resolved.append(chosen)

        if not resolved:
            chosen = _prompt_player_selection(None, roster, str(ri + 1))
            resolved.append(chosen)

        starter = resolved[0]
        subs_with_innings: list[tuple[tuple[str, int | None], int]] = []

        # Read any cached sub innings for this row
        _names_cache_path = cache_dir / "_names.json"
        _row_key = f"r{ri+1:02d}"
        try:
            _nc = json.loads(_names_cache_path.read_text(encoding="utf-8")) if _names_cache_path.exists() else {}
            _cached_sub_innings: list[int] = _nc.get(_row_key, {}).get("sub_innings") or []
        except Exception:
            _cached_sub_innings = []

        _sub_innings_used: list[int] = []

        for pi, sub in enumerate(resolved[1:], 1):
            prev_player = resolved[pi - 1]
            sub_idx = pi - 1  # 0-based index into sub list

            # 1. Use cached value if available
            sub_inning: int | None = _cached_sub_innings[sub_idx] if sub_idx < len(_cached_sub_innings) else None

            if sub_inning:
                click.echo(f"  Row {ri+1}: sub{pi} inning from cache: {sub_inning}")
            else:
                # 2. Try VLM detection (only for the first sub)
                if pi == 1:
                    sub_inning = _detect_sub_inning_vlm(
                        img, ri, row_tops, row_bottoms, col_lefts, n_phys_cols, client, model
                    )
                if sub_inning:
                    click.echo(f"  Row {ri+1}: sub{pi} inning auto-detected: {sub_inning}")
                elif auto_yes:
                    sub_inning = 1
                    click.echo(f"  Row {ri+1}: sub{pi} inning unknown — defaulting to 1 (re-run without -y to set)")
                else:
                    if pi == 1:
                        click.echo(f"  Row {ri+1}: sub inning not auto-detected")
                    sub_inning_str = click.prompt(
                        f"  First inning {sub[0]} batted (replaced {prev_player[0]}, 1-{innings})",
                        default="",
                    )
                    sub_inning = int(sub_inning_str.strip()) if sub_inning_str.strip().isdigit() else 1

            _sub_innings_used.append(sub_inning)
            subs_with_innings.append((sub, sub_inning))

        # Persist sub innings to _names.json so future runs skip the prompt
        if _sub_innings_used:
            try:
                _nc2 = json.loads(_names_cache_path.read_text(encoding="utf-8")) if _names_cache_path.exists() else {}
                _nc2.setdefault(_row_key, {})["sub_innings"] = _sub_innings_used
                _names_cache_path.write_text(json.dumps(_nc2, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        slot_info.append({"starter": starter, "subs": subs_with_innings})

    # Update active_roster to use detected names (used by VLM hints and checks)
    active_roster = [info["starter"] for info in slot_info]  # list of (name, jersey) tuples

    # ── Ground truth ──────────────────────────────────────────────────────────
    if gt_dir:
        gt_root = Path(gt_dir)
    else:
        gt_root = game_dir
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
        [None] * n_phys_cols for _ in range(n_active_rows)
    ]
    to_process: list[tuple[int, int, Path]] = []
    cached_count = 0
    for ri in range(n_active_rows):
        for ci in range(n_phys_cols):
            cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
            if reuse_cache and cf.exists():
                cell = json.loads(cf.read_text(encoding="utf-8"))
                if isinstance(cell.get("result"), str) and cell["result"].strip().lower() == "null":
                    cell["result"] = None
                    cf.write_text(json.dumps(cell, ensure_ascii=False), encoding="utf-8")
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

    total_cells = n_active_rows * n_phys_cols
    click.echo(f"\nCells : {total_cells} total | {cached_count} cached | {len(to_process)} to classify")

    # ── Classify ──────────────────────────────────────────────────────────────
    if to_process:
        t0 = time.monotonic()

        def _process(args: tuple[int, int, Path]) -> tuple[int, int, dict]:
            ri, ci, cf = args
            inning = col_to_inning[ci]
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

    # ── Serial retry sweep for any remaining api_error cells ─────────────────
    # Parallel requests can amplify load and cause 503 bursts. Retry errors one
    # at a time with a longer pause so the API has time to recover.
    error_cells = [
        (ri, ci)
        for ri in range(n_active_rows)
        for ci in range(n_phys_cols)
        if (grid[ri][ci] or {}).get("result") is None
        and ((grid[ri][ci] or {}).get("notes") or "").startswith("api_error:")
    ]
    if error_cells:
        click.echo(f"\n\n-- Retry sweep: {len(error_cells)} api_error cell(s) " + "-" * 30)
        for ri, ci in error_cells:
            inning = col_to_inning[ci]
            name = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
            click.echo(f"  Retrying P{ri+1} ({name}) inn{inning}…")
            y1, y2 = max(0, row_tops[ri]), row_bottoms[ri]
            x1, x2 = max(0, col_lefts[ci]), col_lefts[ci + 1]
            crop = img[y1:y2, x1:x2]
            result = classify_cell(crop, name, inning, client, model)
            cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
            cf.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            grid[ri][ci] = result
            notes = (result.get("notes") or "")
            if result.get("result") is None and notes.startswith("api_error:"):
                click.echo(f"  STILL FAILED: {notes}")
            else:
                click.echo(f"  -> {result.get('result')} (run={result.get('run')})")

        # Abort if any cells are still unresolved after the retry sweep
        still_broken = [
            (ri, ci)
            for ri, ci in error_cells
            if (grid[ri][ci] or {}).get("result") is None
            and ((grid[ri][ci] or {}).get("notes") or "").startswith("api_error:")
        ]
        if still_broken:
            names = [
                f"P{ri+1} inn{col_to_inning[ci]}"
                for ri, ci in still_broken
            ]
            raise click.ClickException(
                f"API errors not resolved after retry: {', '.join(names)}. "
                "Re-run to retry (cached cells will be skipped)."
            )

    # ── Inning wrap detection (post-VLM, pre-rules) ──────────────────────────
    # _layout.json is written after every auto-detection so it can be inspected
    # and edited.  If it already exists (from a prior run or manual edit) it is
    # used as-is; delete the file to force re-detection.
    _layout_path = cache_dir / "_layout.json"
    _layout_from_file = False
    if _layout_path.exists():
        try:
            _lo = json.loads(_layout_path.read_text(encoding="utf-8"))
            _loaded = _lo.get("col_to_inning", [])
            if len(_loaded) == n_phys_cols:
                col_to_inning = _loaded
                _layout_from_file = True
            else:
                click.echo(
                    f"  WARNING: _layout.json has {len(_loaded)} entries but n_phys_cols={n_phys_cols}"
                    " — re-detecting and overwriting."
                )
        except Exception as exc:
            click.echo(f"  WARNING: could not read _layout.json ({exc}) — re-detecting.")

    if not _layout_from_file:
        col_to_inning, overflow_cols = _detect_wrap_from_grid(grid, n_active_rows, n_phys_cols)
        _layout_path.write_text(
            json.dumps({"col_to_inning": col_to_inning}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        overflow_cols = [
            ci + 1  # 1-based
            for ci in range(1, len(col_to_inning))
            if col_to_inning[ci] == col_to_inning[ci - 1]
        ]

    if overflow_cols:
        for oc in overflow_cols:
            inn = col_to_inning[oc - 2]
            src = "file" if _layout_from_file else "auto"
            click.echo(f"WRAP  : inning {inn} continues into column {oc} ({src})")
    else:
        click.echo(f"Wrap  : none detected ({'from file' if _layout_from_file else 'auto'})")

    # ── Auto-detect inning totals if no _totals.txt was found ────────────────
    if gt_totals is None and extra_tops:
        click.echo("\n\n-- Auto-detecting inning totals from scan " + "-" * 23)
        detected_totals = _detect_inning_totals(
            img, extra_tops, extra_bottoms, col_lefts, n_phys_cols,
            col_to_inning, client, model, cache_dir,
        )
        if detected_totals:
            if not auto_yes:
                click.echo(f"  Detected totals for {len(detected_totals)} inning(s) — please review:")
                detected_totals = _interactive_review_totals(detected_totals, innings)
            else:
                click.echo(f"  Detected totals for {len(detected_totals)} inning(s) — skipping review (--yes).")
            _write_totals_txt(game_dir, img_path.stem, detected_totals, innings)
            gt_totals = detected_totals
        else:
            click.echo("  Could not detect totals row — cross-checks will run without GT.")

    # ── Re-read cells where run=True but result=None ─────────────────────────
    n_reread = _reread_run_no_result_cells(
        img, grid, n_active_rows, n_phys_cols,
        row_tops, row_bottoms, col_lefts,
        col_to_inning, row_names,
        client, model, cache_dir,
    )
    if n_reread:
        click.echo(f"  Re-read {n_reread} cell(s) with run=True / result=None")

    # ── Re-read hole cells (null between two non-null in same column) ─────────
    n_holes = _reread_hole_cells(
        img, grid, n_active_rows, n_phys_cols,
        row_tops, row_bottoms, col_lefts,
        col_to_inning, row_names,
        client, model, cache_dir,
    )
    if n_holes:
        click.echo(f"  Re-read {n_holes} hole cell(s)")

    # ── Batting rules (structural post-processing) ───────────────────────────
    click.echo("\n\n-- Batting rules " + "-" * 48)
    last_batter_by_inning = _apply_batting_rules(grid, n_active_rows, n_phys_cols, cache_dir, col_to_inning)

    # ── GT run enforcement (must come after batting rules) ───────────────────
    if gt_totals:
        click.echo("\n\n-- GT run enforcement " + "-" * 43)
        _enforce_gt_runs(grid, n_active_rows, n_phys_cols, gt_totals, cache_dir, col_to_inning)

    # ── Integrity checks ──────────────────────────────────────────────────────
    click.echo("\n\n-- Per-player (row) check " + "-" * 40)
    for ri in range(n_active_rows):
        info = slot_info[ri] if ri < len(slot_info) else None
        if not info:
            name = active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
            _check_row(ri + 1, name, [grid[ri][ci] or {} for ci in range(n_phys_cols)], gt_stats)
            continue
        # Starter + any subs, each gets cells only for the innings they played.
        all_players = [(info["starter"], 0)] + [(p, inn) for p, inn in info["subs"]]
        for pi, ((pname, _j), entry_inn) in enumerate(all_players):
            exit_inn = all_players[pi + 1][1] if pi + 1 < len(all_players) else innings + 1
            player_cells = [
                grid[ri][ci] or {} for ci in range(n_phys_cols)
                if entry_inn <= col_to_inning[ci] < exit_inn
            ]
            prefix = "  ↳ " if pi > 0 else ""
            _check_row(ri + 1, f"{prefix}{pname}", player_cells, gt_stats if pi == 0 else None)

    click.echo("\n\n-- PA sequence check " + "-" * 45)
    _check_pa_sequence(grid, n_active_rows, n_phys_cols, active_roster)

    click.echo("\n\n-- Per-inning (column) check " + "-" * 37)
    from collections import defaultdict as _dd
    _inning_cells_check: dict[int, list[dict]] = _dd(list)
    for ci in range(n_phys_cols):
        inn = col_to_inning[ci]
        for ri in range(n_active_rows):
            _inning_cells_check[inn].append(grid[ri][ci] or {})
    for inn in sorted(_inning_cells_check):
        if inn > innings:
            continue  # buffer columns beyond the game length
        _check_col(inn, _inning_cells_check[inn], gt_totals)

    # ── Run reconciliation against GT inning totals ───────────────────────────
    # Only runs a focused re-check when we extracted FEWER runs than GT expects.
    # Over-counted innings are already handled by _enforce_gt_runs above.
    # Aggregates across overflow columns (multiple cols may map to same inning).
    if gt_totals:
        # Run reconciliation (folded into per-inning section — no separate header)
        from collections import defaultdict as _dd2
        _inning_col_map: dict[int, list[int]] = _dd2(list)
        for ci in range(n_phys_cols):
            _inning_col_map[col_to_inning[ci]].append(ci)

        run_issues: list[str] = []
        for inn in sorted(_inning_col_map):
            if inn not in gt_totals or inn > innings:
                continue
            gt_r = gt_totals[inn]["R"]
            cols = _inning_col_map[inn]
            extracted_r = sum(
                1 for ci in cols for ri in range(n_active_rows)
                if (grid[ri][ci] or {}).get("run")
            )
            if extracted_r == gt_r:
                continue
            if extracted_r > gt_r:
                run_issues.append(f"  Inn {inn}: R={extracted_r} > GT={gt_r} (over-count; should be handled by GT enforcement)")
                continue
            # extracted_r < gt_r: re-examine non-run cells
            run_issues.append(f"  Inn {inn}: R={extracted_r} < GT={gt_r} — rechecking {gt_r - extracted_r} missing run(s)…")
            found = 0
            for ci in cols:
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
                        cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")
                        run_issues.append(f"    -> P{ri+1} {name} inn {inn}: run corrected to True")
                        found += 1
        if run_issues:
            click.echo("\n  -- Run reconciliation:")
            for msg in run_issues:
                click.echo(msg)
            # Re-print per-inning check so corrected run flags are visible
            click.echo("\n  -- Per-inning (column) check after reconciliation:")
            for inn in sorted(_inning_cells_check):
                if inn > innings:
                    continue
                _check_col(inn, _inning_cells_check[inn], gt_totals)
        else:
            click.echo("  Run totals: all match GT.")

    # ── Assemble GameExtraction ───────────────────────────────────────────────
    lineup: list[LineupSlot] = []
    for ri in range(n_active_rows):
        info = slot_info[ri]
        # all_slots: [(player_tuple, entry_inning), ...]; starter entry_inning=0
        all_slots: list[tuple[tuple[str, int | None], int]] = [
            (info["starter"], 0)
        ] + [(p, inn) for p, inn in info["subs"]]

        pa_lists: list[list[PlateAppearance]] = [[] for _ in all_slots]

        for ci in range(n_phys_cols):
            cell = grid[ri][ci] or {}
            r = cell.get("result")
            if r is None:
                continue
            if isinstance(r, str) and r.strip().lower() == "null":
                continue  # string "null" slipped through earlier normalization
            pa = PlateAppearance(
                inning=col_to_inning[ci],
                result=r,
                run_scored=bool(cell.get("run")),
                notes=cell.get("notes") or "",
                rbi=0, sb=0, cs=0,
                confidence=cell.get("confidence", "high"),
            )
            # Assign to last player whose entry_inning <= this inning
            inning = col_to_inning[ci]
            owner_idx = 0
            for idx, (_, entry_inn) in enumerate(all_slots):
                if entry_inn <= inning:
                    owner_idx = idx
            pa_lists[owner_idx].append(pa)

        players_in_slot: list[PlayerEntry] = [
            PlayerEntry(
                name=p_name,
                jersey_number=p_jersey,
                plate_appearances=pas,
                summary=_make_summary(pas),
            )
            for (p_name, p_jersey), pas in zip([p for p, _ in all_slots], pa_lists)
        ]

        lineup.append(LineupSlot(batting_order=ri + 1, players=players_in_slot))

    # ── PA ordering: non-increasing count by batting slot ────────────────────
    order_msgs = _enforce_pa_ordering(lineup)
    if order_msgs:
        click.echo("\n\n-- PA ordering: trimmed phantom PAs " + "-" * 29)
        for m in order_msgs:
            click.echo(m)

    # Inning totals from grid — aggregate across overflow columns
    _n_unique_innings = len(set(col_to_inning))
    runs_per_inning = [0] * _n_unique_innings
    for ci in range(n_phys_cols):
        inn_idx = col_to_inning[ci] - 1  # 0-based
        runs_per_inning[inn_idx] += sum(
            1 for ri in range(n_active_rows) if (grid[ri][ci] or {}).get("run")
        )
    game = GameExtraction(
        game=GameInfo(
            teams={"home": "Quick", "away": opponent or "Unknown"},
            date=date_str,
        ),
        lineup=lineup,
        inning_totals=InningTotals(runs_per_inning=runs_per_inning),
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = game_dir / f"{img_path.stem}_cells.json"
    out_data = game.model_dump()
    out_data["last_batter_by_inning"] = {str(k): v for k, v in last_batter_by_inning.items()}
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(f"\nSaved : {out_path.name}")

    total_pa = sum(len(p.plate_appearances) for s in lineup for p in s.players)
    total_r  = sum(
        sum(1 for pa in p.plate_appearances if pa.run_scored)
        for s in lineup for p in s.players
    )
    click.echo(f"Total : {total_pa} PA extracted   {total_r} runs")
    if gt_totals:
        gt_r = sum(v["R"] for v in gt_totals.values())
        click.echo(f"GT    : {gt_r} runs expected")

    # ── HTML widget ───────────────────────────────────────────────────────────
    try:
        from render_widget import render_widget_for_game
        widget_path = game_dir / f"{img_path.stem}.html"
        render_widget_for_game(out_data, widget_path)
        click.echo(f"Widget: {widget_path.name}")
    except Exception as exc:
        click.echo(f"Widget: skipped ({exc})")

    # ── DB write ──────────────────────────────────────────────────────────────
    if dry_run:
        click.echo("\nDB    : Dry-run — skipping DB write.")
    else:
        from db import get_connection, init_db, find_duplicate_game, delete_game, write_game, _DB_PATH
        init_db(_DB_PATH)
        conn_db = get_connection(_DB_PATH)
        existing = find_duplicate_game(conn_db, date_str, opponent, None)
        if existing is not None:
            click.echo(f"\nDB    : Replacing existing game (id={existing})...")
            delete_game(conn_db, existing)
        game_id = write_game(conn_db, game, str(out_path))
        conn_db.close()
        click.echo(f"DB    : Written as game id={game_id}")

    click.echo(f"Log   : {_log_path.name}")


if __name__ == "__main__":
    main()
