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
from typing import Callable

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
    4 crossbars = home run (HR)  will cover all 4 cells (can look like reverse E with dot in center)
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

VOIDED / MISSCORED CELLS — return null:
  If the BOTTOM-RIGHT quadrant (1st base / result area) is crossed out with
  diagonal lines (an X or two crossing strokes through that quadrant), the PA
  was recorded in the wrong cell and should be ignored → return result: null.
  If ALL FOUR quadrants are crossed out the entire cell is voided → return result: null.
  Do NOT attempt to read a result from a crossed-out or voided cell.

IMPORTANT — unknown or ambiguous out marks:
  Dutch KNBSB scorecards do NOT use appeal plays, "OUT (A)", or any out
  notation not listed above. If you see a circle whose contents are ambiguous
  or do not clearly show a fielder number or DP/SAC/SF, return result: "K".

Also rate your own confidence with two integers, "result_conf" and "run_conf" (1-5).
Ask yourself what the alternative reading would be — don't just report a feeling:
  5  every stroke is unambiguous
  4  clear, with minor doubt on a detail (e.g. 1 vs 2 crossbars, but the count is visible)
  3  plausible, but a specific alternative reading exists — NAME IT in "notes"
  2  you guessed between two readings — name both in "notes"
  1  barely legible
"result_conf" rates how sure you are of "result"; "run_conf" rates how sure you
are of "run" (true OR false — being sure nothing scored is also a high run_conf).

Return ONLY valid JSON — no prose, no explanation, no markdown fences.
For a PA result use the string code. For no PA use JSON null (NOT the string "null").
PA no run:  {"result": "K",  "run": false, "result_conf": 5, "run_conf": 5, "notes": null}
PA + run:   {"result": "1B", "run": true,  "result_conf": 5, "run_conf": 4, "notes": "could be 2 crossbars"}
Empty cell: {"result": null, "run": false, "result_conf": 5, "run_conf": 5, "notes": null}"""


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
    thinking_budget: int | None = None,
) -> str | None:
    """Make one API call with up to 5 retries on transient errors. Returns raw text or 'api_error:...'."""
    import random
    delays = [2, 8, 30, 90]  # seconds between attempts 1→2, 2→3, 3→4, 4→5
    for attempt in range(5):
        try:
            if model.startswith("gemini"):
                from google.genai import types
                cfg_kwargs: dict = dict(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                if thinking_budget is not None:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=media_type),
                        user_text,
                    ],
                    config=types.GenerateContentConfig(**cfg_kwargs),
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


def _coerce_conf(v) -> int | None:
    """Sanitize a result_conf/run_conf value to an int in [1,5], else None
    (missing/invalid values are never fabricated — see #11)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int) and 1 <= v <= 5:
        return v
    if isinstance(v, str) and v.strip().isdigit():
        n = int(v.strip())
        if 1 <= n <= 5:
            return n
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
        if "result_conf" in parsed or "run_conf" in parsed:
            parsed["result_conf"] = _coerce_conf(parsed.get("result_conf"))
            parsed["run_conf"] = _coerce_conf(parsed.get("run_conf"))
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

        # Salvage path 2: PA cell — extract result/run/notes/confidences
        r_m = re.search(r'"result"\s*:\s*(?:"([^"]*?)"|null)', raw)
        run_m = re.search(r'"run"\s*:\s*(true|false)', raw)
        notes_m = re.search(r'"notes"\s*:\s*(?:"([^"]*?)"|null)', raw)
        rc_m = re.search(r'"result_conf"\s*:\s*(\d+)', raw)
        runc_m = re.search(r'"run_conf"\s*:\s*(\d+)', raw)
        if r_m or run_m:
            return {
                "result": (r_m.group(1) or None) if r_m else None,
                "run": run_m.group(1) == "true" if run_m else False,
                "notes": (notes_m.group(1) or None) if notes_m else None,
                "result_conf": _coerce_conf(rc_m.group(1)) if rc_m else None,
                "run_conf": _coerce_conf(runc_m.group(1)) if runc_m else None,
            }
        return None


def _add_adjusted(cell: dict, flag: str) -> None:
    """Record a structural confidence-lowering event on a cell, deduplicated.

    Structural rules re-run on every --reuse-cache pass, so writers must be
    add-if-absent rather than appending unconditionally (see #11 design)."""
    adjusted = cell.setdefault("adjusted", [])
    if flag not in adjusted:
        adjusted.append(flag)


def classify_cell(
    crop: np.ndarray,
    player_name: str,
    inning: int,
    client,
    model: str,
) -> dict:
    """Classify one PA cell. Returns {"result": str|None, "run": bool, "notes": str|None},
    plus "rbi_slot": int|None but only when run=True (see #7 — the key's absence is a
    deliberate sentinel meaning "not applicable / not yet attempted", distinct from a
    present-but-null value meaning "read attempted, no digit found")."""
    img_bytes, media_type = _encode_cell(crop)
    user_text = f"Player: {player_name}  Inning: {inning}\nReturn JSON only."

    for attempt in range(2):  # retry once with stricter prompt on JSON parse failure
        extra = "" if attempt == 0 else " IMPORTANT: output ONLY the JSON object, nothing else."
        system = _CELL_SYSTEM + extra
        raw = _call_api(client, model, img_bytes, media_type, user_text, system, max_tokens=400)
        if raw is None:
            return {"result": None, "run": False, "rbi_slot": None, "notes": "api_error: max retries exceeded"}
        if raw.startswith("api_error:"):
            return {"result": None, "run": False, "rbi_slot": None, "notes": raw}

        parsed = _parse_json_response(raw)
        if parsed is not None:
            if attempt == 1:
                _add_adjusted(parsed, "parse_retry")
            # Focused bottom-left crop for rbi_slot — more reliable than the general prompt.
            # Only set the key when run=True: a run=False cell has nothing to read (#7),
            # and leaving the key absent (rather than None) lets a later pass that flips
            # run to True (hole-reread, HR constraint, GT reconciliation) still trigger
            # _backfill_rbi_cells's "rbi_slot" not in cell check instead of looking already-done.
            if parsed.get("run"):
                parsed["rbi_slot"] = _read_rbi_slot(crop, player_name, inning, client, model)
            return parsed
        if attempt == 0:
            continue
    return {"result": None, "run": False, "rbi_slot": None, "notes": f"parse_error: {raw[:120]}"}


def _read_rbi_slot(
    crop: np.ndarray,
    player_name: str,
    inning: int,
    client,
    model: str,
) -> int | None:
    """Focused read of the bottom-left (home plate) quadrant for an RBI digit.

    Crops to the bottom-left quarter of the cell and asks the VLM whether a
    single handwritten digit 1–9 is present (the batting-slot of the batter
    who drove the run in).  Returns the integer or None.
    """
    h, w = crop.shape[:2]
    bl = crop[h // 2:, :w // 2]
    if bl.size == 0:
        return None
    img_bytes, media_type = _encode_cell(bl, scale=12)
    system = (
        "You are reading the BOTTOM-LEFT quadrant of a Dutch KNBSB baseball scorecard cell. "
        "This quadrant may contain ONE of the following: "
        "  (A) a single batting-order digit 1–9 → the batter who drove in the run (RBI); "
        "  (B) a multi-character notation: SB (stolen base), WP (wild pitch), PB (passed ball), "
        "      CS (caught stealing), or E followed by digits (e.g. E13) → no RBI; "
        "  (C) nothing / blank → no RBI. "
        "Dutch handwritten digits can look unusual: "
        "  • 8 often looks like two overlapping loops or a cursive figure-eight "
        "  • 6 and 9 have a large loop with a tail "
        "  • 1 may look like a simple downstroke or tick "
        "KEY RULE: if you see exactly ONE mark and it could plausibly be a digit 1–9 "
        "(even with cursive/unusual strokes), return that digit. "
        "Return null only when the mark is clearly a multi-character abbreviation (two or more "
        "distinct letters/symbols) or the quadrant is blank. "
        'Examples: {"rbi_slot": 8}  {"rbi_slot": 4}  {"rbi_slot": null} '
        "Return ONLY valid JSON — no prose."
    )
    user_text = (
        f"Player: {player_name}  Inning: {inning}\n"
        "Is there a single digit 1-9, a multi-character notation (SB/WP/PB/E#), or nothing? "
        "Return the digit as an integer, or null."
    )
    raw = _call_api(client, model, img_bytes, media_type, user_text, system,
                    max_tokens=100, thinking_budget=0)
    if not raw or raw.startswith("api_error:"):
        return None
    parsed = _parse_json_response(raw)
    if not isinstance(parsed, dict):
        return None
    val = parsed.get("rbi_slot")
    if isinstance(val, int) and 1 <= val <= 9:
        return val
    if isinstance(val, str) and val.strip().isdigit():
        n = int(val.strip())
        if 1 <= n <= 9:
            return n
    return None


def _read_sb_count(
    crop: np.ndarray,
    player_name: str,
    inning: int,
    client,
    model: str,
) -> int:
    """Count stolen base notations in the three non-result quadrants of a cell.

    Sends the full cell image and asks the VLM to count 'SB' marks in the
    top-left, top-right, and bottom-left quadrants only (bottom-right is the
    result quadrant and never contains SB notations).  Returns 0 or more.
    """
    if crop.size == 0:
        return 0
    img_bytes, media_type = _encode_cell(crop, scale=6)
    system = (
        "You are reading a Dutch KNBSB baseball scorecard cell to count stolen bases. "
        "The cell has a 2×2 quadrant layout. The BOTTOM-RIGHT quadrant is the plate "
        "appearance result — ignore it completely. "
        "Look ONLY at the TOP-LEFT, TOP-RIGHT, and BOTTOM-LEFT quadrants. "
        "Count the number of times 'SB' (stolen base) appears as a handwritten "
        "notation in any of those three quadrants. "
        "Each 'SB' mark counts as one stolen base. There may be 0, 1, 2, or 3. "
        "Do NOT count WP, PB, CS, or any other abbreviation. "
        'Return ONLY valid JSON: {"sb_count": <integer>}'
    )
    user_text = (
        f"Player: {player_name}  Inning: {inning}\n"
        "How many SB notations are in the top-left, top-right, or bottom-left quadrants?"
    )
    raw = _call_api(client, model, img_bytes, media_type, user_text, system,
                    max_tokens=50, thinking_budget=0)
    if not raw or raw.startswith("api_error:"):
        return 0
    parsed = _parse_json_response(raw)
    if not isinstance(parsed, dict):
        return 0
    val = parsed.get("sb_count")
    if isinstance(val, int) and val >= 0:
        return val
    if isinstance(val, str) and val.strip().isdigit():
        return max(0, int(val.strip()))
    return 0


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
        "Look at the CENTER crosshair for a FILLED solid mark or dot, this is a run."
        "Look at the BOTTOM-LEFT quadrant for any mark/digit/letter(s) this is a run."
        "Either indicates a run scored. Respond with exactly: true or exactly: false"
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
                _add_adjusted(cell, "constraint:out->run=False")
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
                # Keep non-structural keys (reread/adjusted/*_conf/rbi_slot) so a
                # future restore (see the cache-load "removed:" handling) doesn't
                # lose the evidence they record.
                removed_cell = dict(grid[ri][ci] or {})
                removed_cell["result"] = None
                removed_cell["run"] = False
                removed_cell["notes"] = f"removed:isolated ({old})"
                grid[ri][ci] = removed_cell
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
                removed_cell = dict(grid[ri][ci] or {})
                removed_cell["result"] = None
                removed_cell["run"] = False
                removed_cell["notes"] = f"removed:after_3_outs ({old})"
                grid[ri][ci] = removed_cell
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
            _slot = row_names[ri] if ri < len(row_names) else {}
            names = _slot.get("players") or []
            _subs = _slot.get("sub_innings") or []
            player_name = names[0] if names else f"P{ri+1}"
            for _k, _si in enumerate(_subs, 1):
                if inn >= _si and _k < len(names):
                    player_name = names[_k]
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
                result["reread"] = "run"
                grid[ri][ci] = result
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                click.echo(f"    → result: {result['result']}  run: {result.get('run')}")
                reread += 1
            else:
                click.echo(f"    → still no result after re-read ({(raw or 'None')[:80]})")
                # result remains null — clear run so the grid stays consistent
                grid[ri][ci]["run"] = False
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")
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
        # Cyclic wrap is only valid when the inning hasn't ended yet (< 3 outs).
        # Once 3 outs are recorded the inning is complete: nulls at the edge of
        # the batting sequence are legitimate non-batters, not holes.  Similarly,
        # overflow columns (same inning as the previous column) never wrap.
        is_overflow = ci > 0 and col_to_inning and col_to_inning[ci] == col_to_inning[ci - 1]
        n_col_outs = sum(
            1 for r in range(n_rows) if _is_out((grid[r][ci] or {}).get("result"))
        )
        use_linear = is_overflow or n_col_outs >= 3
        # Find the first and last non-null row so we can suppress false cyclic holes.
        nonnull_rows = [r for r in range(n_rows) if _has_result(r, ci)]
        min_nonnull = min(nonnull_rows) if nonnull_rows else None
        max_nonnull = max(nonnull_rows) if nonnull_rows else None
        for ri in range(n_rows):
            if _has_result(ri, ci):
                continue
            if use_linear:
                prev_ri = ri - 1
                next_ri = ri + 1
                if prev_ri < 0 or next_ri >= n_rows:
                    continue  # at the edge — legitimate end-of-inning null
            else:
                prev_ri = (ri - 1) % n_rows
                next_ri = (ri + 1) % n_rows
                # Suppress cyclic boundary false positives:
                # If the inning started at P1 (min_nonnull=0), a null at P9
                # wrapping to P1 is just end-of-inning, not a hole.
                if min_nonnull == 0 and ri == n_rows - 1 and next_ri == 0:
                    continue
                # If the inning ended at P9 (max_nonnull=n_rows-1), a null at P1
                # wrapping back from P9 is just before-inning, not a hole.
                if max_nonnull == n_rows - 1 and ri == 0 and prev_ri == n_rows - 1:
                    continue
            if not (_has_result(prev_ri, ci) and _has_result(next_ri, ci)):
                continue
            _slot = row_names[ri] if ri < len(row_names) else {}
            names = _slot.get("players") or []
            _subs = _slot.get("sub_innings") or []
            player_name = names[0] if names else f"P{ri+1}"
            for _k, _si in enumerate(_subs, 1):
                if inn >= _si and _k < len(names):
                    player_name = names[_k]
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
                result["reread"] = "hole"
                grid[ri][ci] = result
                cf.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                click.echo(f"    → result: {result['result']}  run: {result.get('run', False)}")
                reread += 1
            else:
                click.echo(f"    → still unreadable after re-read — leaving null (fix in review)")
                reread += 1
    return reread


def _enforce_pa_ordering(lineup: list) -> list[str]:
    """PA counts must be non-increasing by batting order.

    The lead-off slot gets the most plate appearances; each later slot the same
    or fewer. When a later slot exceeds the previous slot's count, remove the
    excess PAs — lowest-confidence first, then highest inning (most likely a
    phantom from ink bleed or an off-by-one column assignment).
    """
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
                key=lambda x: (x[0].confidence, -x[0].inning),
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

_STAT_COL_WIDTH = 16   # width of one "NAME=val (GT=x)" column in the check tables


def _fmt_stat(name: str, value: int, expected: int | None, label: str = "GT") -> str:
    """Format one stat as a fixed-width column.

    Appends " (GT=x)" (or " (exp=x)") only when an expected value is given AND
    it differs from the extracted value — matching stats print bare so the eye
    goes straight to the annotated ones.
    """
    s = f"{name}={value}"
    if expected is not None and value != expected:
        s += f" ({label}={expected})"
    # Pad to the column width, but always keep at least two spaces before the
    # next column so an unusually long annotation can't run into its neighbour.
    return s.ljust(_STAT_COL_WIDTH - 2) + "  "


def _check_row(slot: int, name: str, cells: list[dict], gt: dict[int, tuple] | None) -> None:
    pa = sum(1 for c in cells if c.get("result") is not None)
    h  = sum(1 for c in cells if _is_hit(c.get("result")))
    ab = sum(1 for c in cells if _is_ab(c.get("result")))
    r  = sum(1 for c in cells if c.get("run"))
    gt_h, gt_ab = gt[slot] if (gt and slot in gt) else (None, None)
    line = (
        f"  [Slot {slot:2d}] {name:<22}  "
        + _fmt_stat("PA", pa, None)
        + _fmt_stat("AB", ab, gt_ab)
        + _fmt_stat("H", h, gt_h)
        + _fmt_stat("R", r, None)
    )
    click.echo(line.rstrip())


def _check_col(inning: int, cells: list[dict], gt: dict[int, dict] | None) -> set[str]:
    """Print the per-inning check line and return which stats still mismatch
    GT ({"R", "H"} — a subset), for #11's confidence scoring to consume."""
    r    = sum(1 for c in cells if c.get("run") and c.get("result") is not None)
    h    = sum(1 for c in cells if _is_hit(c.get("result")))
    outs = sum(1 for c in cells if _is_out(c.get("result")))
    e    = sum(1 for c in cells if re.match(r"^E\d+$", (c.get("result") or "").upper()))
    pa   = sum(1 for c in cells if c.get("result") is not None)
    g = gt[inning] if (gt and inning in gt) else None
    # PA and Outs have no direct GT value: PA is derived (3 outs + R + LOB) and
    # a completed inning always has 3 outs — labelled "exp" to make that clear.
    exp_pa   = 3 + g["R"] + g["LOB"] if g else None
    exp_outs = 3 if g else None
    line = (
        f"  [Inn {inning:<2}]  "
        + _fmt_stat("PA", pa, exp_pa, "exp")
        + _fmt_stat("R", r, g["R"] if g else None)
        + _fmt_stat("H", h, g["H"] if g else None)
        + _fmt_stat("E", e, g["E"] if g else None)
        + _fmt_stat("Outs", outs, exp_outs, "exp")
    )
    click.echo(line.rstrip())
    flags: set[str] = set()
    if g is not None:
        if r != g["R"]:
            flags.add("R")
        if h != g["H"]:
            flags.add("H")
    return flags


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
                _add_adjusted(grid[ri][ci], "gt:run->False")
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")
                click.echo(f"  [GT-R=0] Player {ri+1} inn {inn}: run forced False (GT=0 runs)")
        elif len(extracted) > gt_r:
            click.echo(
                f"  [GT-R] Inn {inn}: {len(extracted)} runs extracted, GT={gt_r} — see run reconciliation"
            )


# ── GT run reconciliation (#1) ────────────────────────────────────────────────

_GT_RUN_REMOVE_GAP = 2   # run_conf gap needed between the last removed and first kept scorer
_RUNNER_OUT_RE = re.compile(
    r"runner (was |got )?(out|retired|caught|thrown out)|caught stealing|\bCS\b|picked off|"
    r"forced? out at|out at (2nd|3rd|home|second|third)",
    re.IGNORECASE,
)


def _reconcile_gt_runs(
    grid: list[list[dict | None]],
    n_rows: int,
    cols_by_inning: dict[int, list[int]],
    gt_totals: dict[int, dict],
    last_batter_by_inning: dict[int, int],
    recheck: "Callable[[int, int, int], bool] | None",
    cache_dir: Path | None,
    names: list[str] | None = None,
) -> list[str]:
    """Make extracted runs per inning match the ground-truth total, both directions (#1).

    Works inning by inning on the batting sequence (rows rotated to the inning's
    lead-off batter, overflow columns appended), because baseball structure gives
    strong priors that a per-cell read does not:

    UNDER-COUNT (extracted R < GT R) — two tiers, then give up loudly:
      1. Forced (structural): runners cannot pass each other, so every batter who
         reached base AHEAD of the last scorer must have scored or been retired on
         the bases. When all 3 outs of the inning are plate-appearance outs (no DP,
         no runner-out note) nobody was retired on the bases, so those runners
         scored. Assigned without a VLM call, tagged "gt:run->True(forced)". If more
         runners are forced than runs are missing, one of the run=True cells is
         probably wrong instead — nothing is assigned, the inning is flagged.
      2. Ranked VLM re-check: remaining non-out, non-scoring cells are re-read via
         ``recheck(ri, ci, inn)`` in order of lowest run_conf first (least sure of
         "no run"), then earliest in the batting sequence (earlier runners score
         first). Only a positive re-read assigns a run ("recheck:run->True").
      3. Still short → listed for review.py; nothing is invented.

    OVER-COUNT (extracted R > GT R, GT R > 0; GT R = 0 is handled earlier by
    ``_enforce_gt_runs``): candidate scorers are ranked by run_conf ascending (a HR
    is never a candidate — the batter always scores). A scorer that sits BEHIND a
    non-scoring runner in an airtight inning is structurally suspect and ranked one
    notch lower. The lowest ``m`` are removed only when there is a clear gap
    (``_GT_RUN_REMOVE_GAP``) between the last removed and the first kept, or when
    every non-HR scorer must go for the inning to add up. Otherwise flagged.

    Mutates ``grid``, writes touched cells to ``cache_dir`` (if given) and returns
    the messages to print. ``recheck`` may be None (no VLM available) — tier 2 is
    then skipped. Pure apart from grid/cache mutation, so it is unit-testable with a
    fake ``recheck``.
    """
    msgs: list[str] = []

    def _name(ri: int) -> str:
        return names[ri] if names and ri < len(names) else f"P{ri+1}"

    def _save(ri: int, ci: int) -> None:
        if cache_dir is not None:
            cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
            cf.write_text(json.dumps(grid[ri][ci], ensure_ascii=False), encoding="utf-8")

    def _run_conf(cell: dict, inn: int) -> int:
        return _score_cell(cell, inn, None)[1]

    for inn in sorted(cols_by_inning):
        if inn not in gt_totals:
            continue
        gt_r = gt_totals[inn]["R"]
        cols = cols_by_inning[inn]

        # Batting sequence for this inning: rotate rows to the lead-off batter.
        # Inning 1 starts at row 0; otherwise the batter after the previous inning's
        # last batter (1-based slot in last_batter_by_inning). Unknown start (an
        # earlier inning had unreadable cells) disables the structural tier.
        if inn == 1:
            start_ri, start_known = 0, True
        elif (inn - 1) in last_batter_by_inning:
            start_ri, start_known = last_batter_by_inning[inn - 1] % n_rows, True
        else:
            start_ri, start_known = 0, False
        seq: list[tuple[int, int, int, dict]] = []   # (pos, ri, ci, cell)
        pos = 0
        for ci in cols:
            for k in range(n_rows):
                ri = (start_ri + k) % n_rows
                cell = grid[ri][ci] or {}
                if cell.get("result") is not None:
                    seq.append((pos, ri, ci, cell))
                pos += 1

        scorers = [s for s in seq if s[3].get("run")]
        extracted_r = len(scorers)
        if extracted_r == gt_r:
            continue

        pa_outs = sum(1 for s in seq if _is_out(s[3].get("result")))
        has_dp = any((s[3].get("result") or "").upper() == "DP" for s in seq)
        airtight = start_known and pa_outs >= 3 and not has_dp
        last_scorer_pos = max((s[0] for s in scorers), default=-1)

        # ── Under-count ──────────────────────────────────────────────────────
        if extracted_r < gt_r:
            need = gt_r - extracted_r
            msgs.append(f"  Inn {inn}: R={extracted_r} < GT={gt_r} — {need} missing run(s)")
            eligible = [
                s for s in seq
                if not s[3].get("run") and not _is_out(s[3].get("result"))
            ]

            # Tier 1: structurally forced runs.
            forced: list[tuple[int, int, int, dict]] = []
            if airtight and scorers:
                forced = [
                    s for s in eligible
                    if s[0] < last_scorer_pos and not _RUNNER_OUT_RE.search(s[3].get("notes") or "")
                ]
                if len(forced) > need:
                    msgs.append(
                        f"    !! {len(forced)} runner(s) ahead of the last scorer but only {need} "
                        f"missing — a run=True cell is probably wrong; nothing assigned, review: "
                        + ", ".join(f"P{ri+1} {c.get('result')}" for _, ri, _, c in forced)
                    )
                    forced = []
            for _, ri, ci, cell in forced:
                cell["run"] = True
                _add_adjusted(cell, "gt:run->True(forced)")
                grid[ri][ci] = cell
                _save(ri, ci)
                msgs.append(
                    f"    -> P{ri+1} {_name(ri)} {cell.get('result')}: run forced True "
                    f"(reached base ahead of a scorer; all 3 outs are PA outs)"
                )
            need -= len(forced)
            forced_ids = {id(s[3]) for s in forced}

            # Tier 2: ranked VLM re-check.
            remaining = [s for s in eligible if id(s[3]) not in forced_ids]
            remaining.sort(key=lambda s: (_run_conf(s[3], inn), s[0]))
            if need > 0 and recheck is not None:
                for _, ri, ci, cell in remaining:
                    if need <= 0:
                        break
                    if recheck(ri, ci, inn):
                        cell["run"] = True
                        _add_adjusted(cell, "recheck:run->True")
                        grid[ri][ci] = cell
                        _save(ri, ci)
                        msgs.append(f"    -> P{ri+1} {_name(ri)} {cell.get('result')}: run corrected to True (VLM re-check)")
                        need -= 1

            # Tier 3: give up loudly.
            if need > 0:
                still = [s for s in remaining if not s[3].get("run")]
                cand = ", ".join(
                    f"P{ri+1} {c.get('result')} (run_conf={_run_conf(c, inn)})"
                    for _, ri, _, c in still[:5]
                ) or "none"
                msgs.append(f"    !! still {need} run(s) short — left for review. Candidates: {cand}")
            continue

        # ── Over-count ───────────────────────────────────────────────────────
        m = extracted_r - gt_r
        msgs.append(f"  Inn {inn}: R={extracted_r} > GT={gt_r} — {m} run(s) too many")
        if gt_r == 0:
            continue  # _enforce_gt_runs already cleared these
        nonscoring_ahead = [
            s[0] for s in seq
            if not s[3].get("run") and not _is_out(s[3].get("result"))
        ]
        ranked: list[tuple[int, int, int, int, dict]] = []   # (key, pos, ri, ci, cell)
        for p, ri, ci, cell in scorers:
            if (cell.get("result") or "").upper() == "HR":
                continue  # a HR always scores
            key = _run_conf(cell, inn)
            if airtight and any(q < p for q in nonscoring_ahead):
                key -= 1  # behind a non-scoring runner: structurally suspect
            ranked.append((key, p, ri, ci, cell))
        # Lowest confidence first; among ties the later batter (the one behind
        # non-scorers) is the more likely phantom.
        ranked.sort(key=lambda t: (t[0], -t[1]))

        if len(ranked) < m:
            msgs.append(
                f"    !! only {len(ranked)} removable scorer(s) (HR runs are fixed) — GT total "
                f"contradicts the HR count; review"
            )
            continue
        clear = len(ranked) == m or (ranked[m][0] - ranked[m - 1][0]) >= _GT_RUN_REMOVE_GAP
        if not clear:
            cand = ", ".join(f"P{ri+1} {c.get('result')} (run_conf={k})" for k, _, ri, _, c in ranked[: m + 2])
            msgs.append(f"    !! no clear confidence gap between scorers — left for review. Candidates: {cand}")
            continue
        for _, _, ri, ci, cell in ranked[:m]:
            cell["run"] = False
            _add_adjusted(cell, "gt:run->False")
            grid[ri][ci] = cell
            _save(ri, ci)
            msgs.append(f"    -> P{ri+1} {_name(ri)} {cell.get('result')}: run removed (lowest run_conf, clear gap)")

    return msgs


# ── RBI backfill ─────────────────────────────────────────────────────────────

def _backfill_rbi_cells(
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
    """Re-read run-scoring cells that are missing rbi_slot.

    Runs last in the pipeline (see #7), after every pass that can set
    run=True on a cell — old-format cache entries that pre-date rbi_slot
    tracking, but also any cell whose run flag was only settled to True by
    hole-reread, the HR constraint, or GT reconciliation this same run.
    classify_cell omits the "rbi_slot" key entirely for run=False cells (see
    its docstring), so "key absent" reliably means "never attempted" here —
    only cells with run=True and no rbi_slot key are sent to the VLM; every
    other cell is left untouched. Returns the number of cells updated.
    """
    to_backfill = [
        (ri, ci)
        for ri in range(n_rows)
        for ci in range(n_cols)
        if (grid[ri][ci] or {}).get("run") and "rbi_slot" not in (grid[ri][ci] or {})
    ]
    n_run_cells = sum(
        1 for ri in range(n_rows) for ci in range(n_cols)
        if (grid[ri][ci] or {}).get("run")
    )
    if not to_backfill:
        click.echo(f"\n-- RBI: {n_run_cells} run cell(s) — rbi_slot already read for all (no backfill needed)")
        return 0

    click.echo(
        f"\n\n-- RBI backfill: {len(to_backfill)} run cell(s) missing rbi_slot key " + "-" * 20
    )
    updated = 0
    for ri, ci in to_backfill:
        inn = col_to_inning[ci]
        slot_info = row_names[ri] if ri < len(row_names) else {}
        names = slot_info.get("players") or []
        sub_innings = slot_info.get("sub_innings") or []
        player_name = names[0] if names else f"P{ri+1}"
        for k, si in enumerate(sub_innings, 1):
            if inn >= si and k < len(names):
                player_name = names[k]
        y1, y2 = max(0, row_tops[ri]), row_bottoms[ri]
        x1 = max(0, col_lefts[ci])
        x2 = col_lefts[ci + 1] if ci + 1 < len(col_lefts) else img.shape[1]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        click.echo(f"  P{ri+1} ({player_name}) inn {inn}: reading bottom-left for rbi_slot…")
        # Keep existing result/run — only add rbi_slot via focused bottom-left read
        existing = dict(grid[ri][ci] or {})
        rbi_slot = _read_rbi_slot(crop, player_name, inn, client, model)
        existing["rbi_slot"] = rbi_slot
        grid[ri][ci] = existing
        cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
        cf.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        click.echo(f"    → rbi_slot={rbi_slot}")
        updated += 1
    return updated


def _backfill_sb_cells(
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
    """Read sb_count for reached-base cells missing the key (old-format cache).

    Targets cells where the batter reached base (result is not null and not an
    out) and the 'sb_count' key is absent.  Sends the full cell to the VLM and
    counts SB notations in the three non-result quadrants.
    Returns the number of cells updated.
    """
    to_backfill = [
        (ri, ci)
        for ri in range(n_rows)
        for ci in range(n_cols)
        if (
            (grid[ri][ci] or {}).get("result") is not None
            and not _is_out((grid[ri][ci] or {}).get("result"))
            and "sb_count" not in (grid[ri][ci] or {})
        )
    ]
    if not to_backfill:
        return 0

    click.echo(
        f"\n\n-- SB backfill: {len(to_backfill)} reached-base cell(s) missing sb_count " + "-" * 20
    )
    updated = 0
    for ri, ci in to_backfill:
        inn = col_to_inning[ci]
        slot_info = row_names[ri] if ri < len(row_names) else {}
        names = slot_info.get("players") or []
        sub_innings = slot_info.get("sub_innings") or []
        player_name = names[0] if names else f"P{ri+1}"
        for k, si in enumerate(sub_innings, 1):
            if inn >= si and k < len(names):
                player_name = names[k]
        y1, y2 = max(0, row_tops[ri]), row_bottoms[ri]
        x1 = max(0, col_lefts[ci])
        x2 = col_lefts[ci + 1] if ci + 1 < len(col_lefts) else img.shape[1]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        click.echo(f"  P{ri+1} ({player_name}) inn {inn}: counting SB…")
        existing = dict(grid[ri][ci] or {})
        sb_count = _read_sb_count(crop, player_name, inn, client, model)
        existing["sb_count"] = sb_count
        grid[ri][ci] = existing
        cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
        cf.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        click.echo(f"    → sb_count={sb_count}")
        updated += 1
    return updated


def _enforce_constraints(
    grid: list[list[dict | None]],
    n_rows: int,
    n_cols: int,
    cache_dir: Path,
) -> int:
    """Enforce logical consistency rules on classified cells.

    Applied in order:
      1. E (error) → never an out. If VLM somehow returned an out-looking result
         alongside run=True and notes suggest error, the run is preserved.
         More concretely: any result matching ^E\\d* is safe (not an out).
      2. Out → run must be False. A retired batter cannot have scored.
      3. HR → run must be True (the batter always scores on his own home run).
    """
    changed = 0
    for ri in range(n_rows):
        for ci in range(n_cols):
            cell = grid[ri][ci]
            if not cell:
                continue
            result = (cell.get("result") or "").upper().strip()
            run = bool(cell.get("run"))
            dirty = False

            # Rule 1: E# (error) is never an out. _is_out already returns False
            # for errors, but if the result was mis-read as an out result yet the
            # cell also has run=True (contradictory), clear the out interpretation
            # by keeping run as-is and leaving result untouched.
            # This rule is a no-op when result correctly says "E" — it only matters
            # as a guard before rule 2.
            is_error = bool(re.match(r"^E\d*$", result))

            # Rule 2: out → run=False (skip if this cell is an error — rule 1 takes precedence)
            if not is_error and _is_out(result) and run:
                cell = dict(cell)
                cell["run"] = False
                _add_adjusted(cell, "constraint:out->run=False")
                notes = (cell.get("notes") or "").strip()
                cell["notes"] = (notes + " [constraint: out→run=False]").strip()
                grid[ri][ci] = cell
                dirty = True
                click.echo(
                    f"  [constraint] P{ri+1} c{ci+1} ({result}+run=True) → run forced False"
                )
                changed += 1

            # Rule 3: HR → run=True (the batter always scores on his own home run)
            if result == "HR" and not run:
                cell = dict(cell)
                cell["run"] = True
                _add_adjusted(cell, "constraint:HR->run=True")
                notes = (cell.get("notes") or "").strip()
                cell["notes"] = (notes + " [constraint: HR→run=True]").strip()
                grid[ri][ci] = cell
                dirty = True
                click.echo(f"  [constraint] P{ri+1} c{ci+1} (HR) → run forced True")
                changed += 1

            if dirty:
                cf = cache_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"
                cf.write_text(json.dumps(cell, ensure_ascii=False), encoding="utf-8")
    return changed


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
    existing_bytes = players_txt.read_bytes() if players_txt.exists() else b""
    with open(players_txt, "a", encoding="utf-8") as f:
        if existing_bytes and existing_bytes[-1:] != b"\n":
            f.write("\n")
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
    h   = sum(1 for pa in pas if _is_hit(pa.result))
    ab  = sum(1 for pa in pas if _is_ab(pa.result))
    r   = sum(1 for pa in pas if pa.run_scored)
    rbi = sum(pa.rbi for pa in pas)
    return PASummary(PA=len(pas), AB=ab, H=h, R=r, RBI=rbi)


_AMBIGUOUS_NOTES_RE = re.compile(r"ambig|unclear|guess|possibl|faint", re.IGNORECASE)


def _score_cell(
    cell: dict,
    inning: int,
    inning_flags: dict[int, set[str]] | None,
) -> tuple[int, int, list[str]]:
    """Derive (result_conf, run_conf, reasons) for one cell (see #11).

    Starts from the VLM's own result_conf/run_conf self-report and adjusts it
    with structural evidence gathered elsewhere in the pipeline: rereads,
    rule corrections ("adjusted" flags), a run without a corroborating RBI
    digit, and any inning-level GT mismatch that survives every enforcement
    pass. Floors at 1, caps at 5.
    """
    reasons: list[str] = []

    result_conf = cell.get("result_conf")
    run_conf = cell.get("run_conf")
    if result_conf is None or run_conf is None:
        reasons.append("legacy")
    if result_conf is None:
        result_conf = 4
    if run_conf is None:
        run_conf = 4

    reread = cell.get("reread")
    if reread:
        result_conf -= 1
        run_conf -= 1
        reasons.append(f"reread:{reread}")

    for flag in cell.get("adjusted") or []:
        run_conf -= 1
        reasons.append(flag)

    if cell.get("run") and cell.get("rbi_slot") is None:
        run_conf -= 1
        reasons.append("run_without_rbi_digit")

    flags = (inning_flags or {}).get(inning) or set()
    if "R" in flags:
        run_conf -= 1
        reasons.append("inning_R_mismatch")
    if "H" in flags and _is_hit(cell.get("result")):
        result_conf -= 1
        reasons.append("inning_H_mismatch")

    if _AMBIGUOUS_NOTES_RE.search(cell.get("notes") or ""):
        result_conf -= 1
        reasons.append("ambiguous_notes")

    result_conf = max(1, min(5, result_conf))
    run_conf = max(1, min(5, run_conf))
    return result_conf, run_conf, reasons


def _check_rbi_leq_runs(slot_data: list[tuple]) -> list[str]:
    """Per-inning invariant (#7): RBI credited can never exceed runs scored.

    An RBI is only ever credited alongside a run=True cell (see phases 1/2
    above — self-RBI on a HR, or an rbi_slot digit on a *different* run=True
    cell), so each run can contribute at most one RBI credit. A violation
    here means a bug in the attribution logic, not a scorekeeping quirk —
    it is not auto-corrected, just surfaced alongside the other RBI warnings.
    """
    rbi_by_inning: dict[int, int] = {}
    runs_by_inning: dict[int, int] = {}
    for _, pa_lists in slot_data:
        for pas in pa_lists:
            for pa in pas:
                rbi_by_inning[pa.inning] = rbi_by_inning.get(pa.inning, 0) + pa.rbi
                if pa.run_scored:
                    runs_by_inning[pa.inning] = runs_by_inning.get(pa.inning, 0) + 1
    warnings_out: list[str] = []
    for inn in sorted(set(rbi_by_inning) | set(runs_by_inning)):
        rbi = rbi_by_inning.get(inn, 0)
        runs = runs_by_inning.get(inn, 0)
        if rbi > runs:
            warnings_out.append(
                f"  WARNING: inn {inn}: RBI total {rbi} > runs scored {runs} "
                "— RBI can never exceed runs"
            )
    return warnings_out


def _build_slot_data(
    grid: list[list[dict | None]],
    slot_info: list[dict],
    n_active_rows: int,
    n_phys_cols: int,
    col_to_inning: list[int],
    inning_flags: dict[int, set[str]] | None = None,
) -> tuple[list[tuple], list[str]]:
    """Turn the cell grid into per-slot PlateAppearance lists with RBIs attributed.

    Returns ``(slot_data, warnings)`` where ``slot_data[ri] == (all_slots, pa_lists)``:
    ``all_slots`` is ``[(player_tuple, entry_inning), ...]`` (starter has entry 0) and
    ``pa_lists[k]`` holds the PlateAppearances of the k-th player in that slot.

    RBI attribution rules:
      * A HR credits its batter with 1 RBI at construction (they drove themselves in).
      * Every run=True cell carrying an ``rbi_slot`` digit credits the batter in that
        slot with 1 RBI on their PA in the same inning.
      * If ``rbi_slot`` points at the runner's OWN slot, no extra RBI is added: for a
        HR that RBI is already counted (scorers write the batter's own number in the
        bottom-left of a HR cell), and for any other result it is impossible — a
        batter cannot drive themselves in without a HR — so it is reported instead.
        This self-reference was the cause of the 3-RBI-on-a-2-RBI-HR bug (#12).

    ``inning_flags`` (optional; see #11 / ``_score_cell``) maps inning number to
    the set of stats ({"R", "H"}) still mismatching GT after every pass — used
    to dock confidence on cells in an inning whose totals still don't add up.
    """
    slot_data: list[tuple] = []
    warnings_out: list[str] = []

    # Phase 1: build PA lists for every slot (HR self-RBI only).
    for ri in range(n_active_rows):
        info = slot_info[ri]
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
            inning = col_to_inning[ci]
            result_conf, run_conf, conf_reasons = _score_cell(cell, inning, inning_flags)
            pa = PlateAppearance(
                inning=inning,
                result=r,
                run_scored=bool(cell.get("run")),
                notes=cell.get("notes") or "",
                rbi=1 if (r or "").upper() == "HR" else 0,
                sb=int(cell.get("sb_count") or 0),
                cs=0,
                confidence=min(result_conf, run_conf),
                result_conf=result_conf,
                run_conf=run_conf,
                conf_reasons=conf_reasons,
            )
            owner_idx = 0
            for idx, (_, entry_inn) in enumerate(all_slots):
                if entry_inn <= inning:
                    owner_idx = idx
            pa_lists[owner_idx].append(pa)

        slot_data.append((all_slots, pa_lists))

    # Phase 2: attribute RBIs from rbi_slot digits on run cells.
    for ri_runner in range(n_active_rows):
        for ci in range(n_phys_cols):
            cell = grid[ri_runner][ci] or {}
            if not cell.get("run") or cell.get("result") is None:
                continue
            rbi_slot = cell.get("rbi_slot")
            if not isinstance(rbi_slot, int) or not 1 <= rbi_slot <= n_active_rows:
                continue
            ri_batter = rbi_slot - 1
            inning = col_to_inning[ci]
            if ri_batter == ri_runner:
                if (cell.get("result") or "").upper() != "HR":
                    warnings_out.append(
                        f"  P{ri_runner+1} inn {inning}: rbi_slot={rbi_slot} points at the runner's "
                        f"own slot on a {cell.get('result')} — impossible, RBI not credited"
                    )
                continue  # HR self-RBI already counted in phase 1
            all_slots_b, pa_lists_b = slot_data[ri_batter]
            owner_idx_b = 0
            for idx, (_, entry_inn) in enumerate(all_slots_b):
                if entry_inn <= inning:
                    owner_idx_b = idx
            target_pas = pa_lists_b[owner_idx_b]
            credited = False
            for pa in target_pas:
                if pa.inning == inning:
                    pa.rbi += 1
                    credited = True
                    break
            if not credited and target_pas:
                target_pas[-1].rbi += 1

    warnings_out.extend(_check_rbi_leq_runs(slot_data))
    return slot_data, warnings_out


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
@click.option("--reset-names", is_flag=True, default=False,
              help="Delete the _names.json cache before running so player names are re-detected from scratch.")
@click.option("--gt-dir", default=None,
              help="Ground-truth directory (default: the game folder inside data_root/games/).")
@click.option("--left-skip", "left_skip_frac", default=0.05, show_default=True, type=float,
              help="Fraction of image width to skip before V-line detection (skips player-info area). "
                   "Increase to ~0.30–0.35 for landscape/low-res scans with wide player-info columns.")
@click.option("--grid-start", default=None, type=int,
              help="X pixel of the first inning's left edge. Bypasses V-line detection entirely "
                   "when combined with --grid-width.")
@click.option("--grid-width", default=None, type=int,
              help="Pixel width of each inning column. Use with --grid-start to force a uniform grid.")
@click.option("--cell-height", "cell_height", default=None, type=int,
              help="Override detected row height in pixels. Use when bimodal H-line detection gives wrong cell_size.")
def main(
    image_path, players_file, active_players, innings,
    n_player_rows, model, workers,
    reuse_cache, dry_run, auto_yes, gt_dir, left_skip_frac,
    grid_start, grid_width, cell_height, reset_names,
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
        n_inning_cols=innings,
        left_skip_frac=left_skip_frac,
        grid_start=grid_start,
        grid_col_width=grid_width,
        cell_height=cell_height,
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
    if reset_names:
        _nc_path = cache_dir / "_names.json"
        if _nc_path.exists():
            _nc_path.unlink()
            click.echo("  --reset-names: deleted _names.json cache")
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
                # Sub slot with short unmatched name → likely VLM noise, skip.
                # Real abbreviated names are at least "A. X" (4 significant chars).
                if pi > 0 and len(detected.replace(".", "").replace(" ", "")) < 4:
                    click.echo(f"  Row {ri+1} sub{pi}: '{detected}' too short to match, ignoring.")
                    break
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
                # Clear stale hole-fallback:BB cells — the hole detection is now
                # more accurate, so these should be re-evaluated as true nulls.
                if "hole-fallback:BB" in notes:
                    cell = {"result": None, "run": False, "rbi_slot": None, "notes": None}
                    cf.write_text(json.dumps(cell, ensure_ascii=False), encoding="utf-8")
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
                        # Keep every other key (reread/adjusted/*_conf/rbi_slot) —
                        # they describe VLM reads, not the rule's removal state.
                        cell = dict(cell)
                        cell["result"] = None if orig in ("null", "None") else orig
                        cell["run"] = False
                        cell["notes"] = None
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

    # ── SB backfill: count stolen bases for reached-base cells ────────────────
    n_sb_backfill = _backfill_sb_cells(
        img, grid, n_active_rows, n_phys_cols,
        row_tops, row_bottoms, col_lefts,
        col_to_inning, row_names,
        client, model, cache_dir,
    )
    if n_sb_backfill:
        click.echo(f"  SB backfill: updated {n_sb_backfill} cell(s)")

    # ── Re-read hole cells (null between two non-null in same column) ─────────
    n_holes = _reread_hole_cells(
        img, grid, n_active_rows, n_phys_cols,
        row_tops, row_bottoms, col_lefts,
        col_to_inning, row_names,
        client, model, cache_dir,
    )
    if n_holes:
        click.echo(f"  Re-read {n_holes} hole cell(s)")

    # ── Logical constraint enforcement ────────────────────────────────────────
    n_constraints = _enforce_constraints(grid, n_active_rows, n_phys_cols, cache_dir)
    if n_constraints:
        click.echo(f"  Enforced {n_constraints} constraint violation(s)")

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
    # inning_flags feeds #11's confidence scoring (_score_cell): which stats
    # still mismatch GT for a given inning, after every enforcement pass has
    # had a chance to fix it. Overwritten below if reconciliation re-checks.
    inning_flags: dict[int, set[str]] = {}
    for inn in sorted(_inning_cells_check):
        if inn > innings:
            continue  # buffer columns beyond the game length
        inning_flags[inn] = _check_col(inn, _inning_cells_check[inn], gt_totals)

    # ── Run reconciliation against GT inning totals (#1) ─────────────────────
    # Both directions: structurally forced runs + ranked VLM re-check when we
    # extracted FEWER runs than GT, gated removal of the least-confident scorer
    # when we extracted MORE. See _reconcile_gt_runs for the rules.
    # Aggregates across overflow columns (multiple cols may map to same inning).
    if gt_totals:
        from collections import defaultdict as _dd2
        _inning_col_map: dict[int, list[int]] = _dd2(list)
        for ci in range(n_phys_cols):
            if col_to_inning[ci] <= innings:
                _inning_col_map[col_to_inning[ci]].append(ci)

        _row_names_flat = [
            active_roster[ri][0] if ri < len(active_roster) else f"P{ri+1}"
            for ri in range(n_active_rows)
        ]

        def _recheck_cell(ri: int, ci: int, inn: int) -> bool:
            y1, y2 = max(0, row_tops[ri]), row_bottoms[ri]
            x1 = max(0, col_lefts[ci])
            x2 = col_lefts[ci + 1] if ci + 1 < len(col_lefts) else img.shape[1]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return False
            return _recheck_run(crop, _row_names_flat[ri], inn, client, model)

        run_issues = _reconcile_gt_runs(
            grid, n_active_rows, dict(_inning_col_map), gt_totals,
            last_batter_by_inning, _recheck_cell, cache_dir, _row_names_flat,
        )
        if run_issues:
            click.echo("\n  -- Run reconciliation:")
            for msg in run_issues:
                click.echo(msg)
            # Re-print per-inning check so corrected run flags are visible
            click.echo("\n  -- Per-inning (column) check after reconciliation:")
            for inn in sorted(_inning_cells_check):
                if inn > innings:
                    continue
                inning_flags[inn] = _check_col(inn, _inning_cells_check[inn], gt_totals)
        else:
            click.echo("  Run totals: all match GT.")

    # ── RBI backfill: re-read every cell that ended up run=True but never got
    # an rbi_slot read (#7). Deliberately runs LAST, after every pass that can
    # flip run False→True (hole-reread, HR constraint, GT reconciliation) —
    # rbi_slot is only ever meaningful once run has settled to its final value.
    n_rbi_backfill = _backfill_rbi_cells(
        img, grid, n_active_rows, n_phys_cols,
        row_tops, row_bottoms, col_lefts,
        col_to_inning, row_names,
        client, model, cache_dir,
    )
    if n_rbi_backfill:
        click.echo(f"  RBI backfill: updated {n_rbi_backfill} cell(s)")

    # ── Assemble GameExtraction ───────────────────────────────────────────────
    # Phases 1+2 (PA lists + RBI attribution) live in _build_slot_data so they
    # can be unit-tested without an image or API client.
    slot_data, rbi_warnings = _build_slot_data(
        grid, slot_info, n_active_rows, n_phys_cols, col_to_inning, inning_flags
    )
    if rbi_warnings:
        click.echo("\n\n-- RBI attribution warnings " + "-" * 38)
        for m in rbi_warnings:
            click.echo(m)

    # Phase 3: build LineupSlots from the completed PA lists.
    lineup: list[LineupSlot] = []
    for ri, (all_slots, pa_lists) in enumerate(slot_data):
        players_in_slot: list[PlayerEntry] = [
            PlayerEntry(
                name=p_name,
                jersey_number=p_jersey,
                innings_played=",".join(str(i) for i in sorted({pa.inning for pa in pas})) or None,
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
        debug_img_path = game_dir / f"{img_path.stem}_grid_debug.png"
        render_widget_for_game(out_data, widget_path, debug_img_path=debug_img_path)
        click.echo(f"Widget: {widget_path.name}")
        os.startfile(str(widget_path))
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
