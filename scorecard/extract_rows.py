#!/usr/bin/env python3
"""
extract_rows.py - Extract a full game from per-row crop images.

Steps:
  1. Run step-1 (visual scan) on each row image individually.
  2. Concatenate all descriptions into one document.
  3. Run step-2 (JSON parsing) on the combined description.
  4. Validate and print the result.

Usage:
  uv run python extract_rows.py <rows_dir> [--model ...] [--dry-run]
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

import anthropic
import click
from dotenv import load_dotenv
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from db import get_connection, init_db, find_duplicate_game, delete_game, write_game
from export_season import export_season
from extract import _STEP1_SYSTEM, _STEP2_SYSTEM, _make_image_content, _remap_summary_keys
from models import GameExtraction
from split_rows import split_from_red_lines, split_scorecard


_TOTALS_SYSTEM = (
    "You read ONLY the bottom team-totals rows of a KNBSB baseball scorecard. "
    "The strip shows the inning-number header on top (innings 1..9) and, below, "
    "two total rows.\n"
    "ROW 'Totaal per inning': each inning is a 2x2 cell split by a crosshair. The "
    "four quadrants hold, per inning:\n"
    "  TOP-LEFT  = fielding ERRORS that inning\n"
    "  TOP-RIGHT = HITS that inning\n"
    "  BOT-LEFT  = runners LEFT ON BASE that inning\n"
    "  BOT-RIGHT = RUNS scored that inning\n"
    "ROW 'Totaal': the CUMULATIVE running team run total after each inning (it never "
    "decreases left to right).\n"
    "Read innings 1 through 9, left to right. A blank quadrant/cell is 0.\n"
    "CRITICAL: Output ONLY the five lines below. No analysis, no prose, no markdown — "
    "just the lines, numbers only, space-separated:\n"
    "RUNS: n n n n n n n n n\n"
    "HITS: n n n n n n n n n\n"
    "ERRORS: n n n n n n n n n\n"
    "LOB: n n n n n n n n n\n"
    "CUMULATIVE: n n n n n n n n n\n"
    "For any row you genuinely cannot read, write its value as: NONE"
)


def _write_totals_cache(path, card: dict, innings: int) -> None:
    """Write the card per-inning totals as a simple, hand-editable table."""
    runs = card.get("runs_per_inning") or []
    hits = card.get("hits") or []
    errors = card.get("errors") or []
    lob = card.get("lob") or []

    def g(lst: list[int], i: int) -> int:
        return lst[i] if i < len(lst) else 0

    lines = [
        "# Totaal per inning — hand-edit to match the scorecard, then re-run.",
        "# This is the ground-truth anchor for all cross-checks/enforcement.",
        "# One row per inning. Whitespace-separated columns:",
        "#   inning  runs  hits  errors  lob",
        "#   runs=BOT-RIGHT  hits=TOP-RIGHT  errors=TOP-LEFT  lob=BOT-LEFT",
    ]
    for i in range(innings):
        lines.append(f"{i + 1}\t{g(runs, i)}\t{g(hits, i)}\t{g(errors, i)}\t{g(lob, i)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_totals_cache(path, innings: int) -> dict:
    """Read the hand-editable per-inning totals table into a card dict."""
    runs = [0] * innings
    hits = [0] * innings
    errors = [0] * innings
    lob = [0] * innings
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", s)]
        if len(nums) >= 5 and 1 <= nums[0] <= innings:
            i = nums[0] - 1
            runs[i], hits[i], errors[i], lob[i] = nums[1], nums[2], nums[3], nums[4]
    return {"runs_per_inning": runs, "runs": runs, "hits": hits,
            "errors": errors, "lob": lob, "cumulative": None}


_STATS_SYSTEM = (
    "This strip is the far-right PER-PLAYER TOTALS column of a baseball scorecard, "
    "one row per batting-order slot, top to bottom = slots 1, 2, 3 ... down the order. "
    "Each cell shows the player's line as \"H-AB\" (hits - at-bats), e.g. '3-4' = 3 hits, "
    "4 at-bats. Some slots have TWO stacked values (a substitution) — report both, "
    "starter first. Output ONLY lines of the form 'slot H AB' (whitespace separated), "
    "no prose. A substitution slot produces two lines with the same slot number."
)


_SECOND_PASS_SYSTEM = """You are re-reading ONE player's row of a KNBSB baseball scorecard to CORRECT the
plate-appearance results. The image shows the inning-number header on top and the
player's scoring row below. For each inning the player batted, read the BOT-RIGHT
result and whether the batter scored.

CELL RULES (2x2 split by a red crosshair):
- A LARGE circle filling BOT-RIGHT = OUT; read contents (K, KL, F7, 6-3, 5-4, 3...).
- A circle ON A BASE (top quadrant / centre) = a baserunning out, NOT the result.
- Vertical stroke with crossbars = HIT; base from X marks in the top quadrants:
  X_COUNT 0 -> 1B, 1 -> 2B, 2 -> 3B.
- Letters "BB" (two rounded humps) NOT inside a circle = walk (BB).
- "E"+digit = reached on error (E6); "FC" = fielder's choice; "HP"/"HBP" = HBP.
- Run scored if BOT-LEFT has any mark OR a black dot sits on the centre crosshair.

Use the CONSTRAINTS to disambiguate hard cells (they are ground truth from the
card's printed totals):
{constraints}

Output ONLY one line per inning listed above, nothing else:
  <inning> | <result> | <run 0 or 1>
e.g.  4 | 1B | 0
"""


def _build_refine_context(player: dict, h_target: int | None, walk_target: int | None,
                          card: dict | None) -> str:
    innings = sorted(int(pa.get("inning", 0)) for pa in player.get("plate_appearances", []))
    lines = [f"- This player batted in innings: {innings}."]
    if h_target is not None:
        lines.append(f"- This player has exactly {h_target} hit(s) total (1B/2B/3B/HR).")
    if walk_target is not None and walk_target >= 0:
        lines.append(f"- This player has exactly {walk_target} walk(s)/HBP total (BB/HBP).")
    if card:
        ch, ce, cr = card.get("hits") or [], card.get("errors") or [], card.get("runs_per_inning") or []
        for i in innings:
            parts = []
            if i - 1 < len(cr):
                parts.append(f"{cr[i-1]} run(s)")
            if i - 1 < len(ch):
                parts.append(f"{ch[i-1]} hit(s)")
            if i - 1 < len(ce):
                parts.append(f"{ce[i-1]} error(s)")
            if parts:
                lines.append(f"- Inning {i} (whole team): " + ", ".join(parts) + ".")
    return "\n".join(lines)


def _apply_refinement(player: dict, text: str) -> list[str]:
    by_inn = {int(pa.get("inning", 0)): pa for pa in player.get("plate_appearances", [])}
    changes: list[str] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            inn = int(parts[0])
            result = parts[1]
            run = len(parts) >= 3 and parts[2].strip() in ("1", "true", "True", "yes")
            pa = by_inn.get(inn)
            if pa and result and pa.get("result") != result:
                changes.append(f"inn{inn} {pa.get('result')}->{result}")
                pa["result"] = result
                pa["run_scored"] = run
                pa["confidence"] = "high"
            elif pa:
                pa["run_scored"] = run
    return changes


def _crop_stats_strip(image_path, out_dir, left_ratio: float = 0.85, upscale: int = 3):
    """Crop the far-right per-player H-AB totals column and upscale it."""
    from PIL import Image as _Image
    img_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    im = _Image.open(img_path).convert("RGB")
    w, h = im.size
    crop = im.crop((int(w * left_ratio), 0, w, h))
    if upscale > 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale), _Image.LANCZOS)
    out = out_dir / f"{img_path.stem}_stats.png"
    crop.save(out, "PNG")
    return out


def _parse_player_stats(text: str) -> list[tuple[int, int, int]]:
    """Parse 'slot H AB' lines into ordered (slot, H, AB) tuples."""
    rows: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        nums = [int(x) for x in re.findall(r"-?\d+", line)]
        if len(nums) >= 3:
            rows.append((nums[0], nums[1], nums[2]))
    return rows


def _write_player_stats_cache(path, rows: list[tuple[int, int, int]]) -> None:
    lines = [
        "# Per-player totals — hand-edit to match the scorecard, then re-run.",
        "# H = hits, AB = at-bats. One row per player; a substitution slot has two",
        "# rows with the same slot number (starter first).",
        "#   slot  H  AB",
    ]
    for slot, h, ab in rows:
        lines.append(f"{slot}\t{h}\t{ab}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_player_stats_cache(path) -> dict[int, list[tuple[int, int]]]:
    """Return {slot: [(H, AB), ...]} preserving starter-then-sub order."""
    by_slot: dict[int, list[tuple[int, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", s)]
        if len(nums) >= 3:
            by_slot.setdefault(nums[0], []).append((nums[1], nums[2]))
    return by_slot


def _enforce_player_hits(raw_json: dict, stats_by_slot: dict[int, list[tuple[int, int]]],
                         low_conf_only: bool = True) -> list[str]:
    """Force each player's hit count to the card's per-player H. More precise than
    the per-inning hit pass because it pins WHO (just maybe the wrong cell/type).
    Promotes a non-hit PA to 1B / demotes a hit to Out; low-confidence first.
    """
    msgs: list[str] = []
    HITS = {"1B", "2B", "3B", "HR"}

    def is_low(pa):
        return pa.get("confidence", "high") == "low"

    for slot in raw_json.get("lineup", []):
        bo = slot.get("batting_order")
        targets = stats_by_slot.get(bo, [])
        for idx, p in enumerate(slot.get("players", [])):
            if idx >= len(targets):
                continue
            h_target = targets[idx][0]
            pas = p.get("plate_appearances", [])
            hits = [pa for pa in pas if (pa.get("result") or "").upper() in HITS]
            cur = len(hits)
            if cur == h_target:
                continue
            name = p.get("name", f"slot{bo}")
            if cur > h_target:
                cand = sorted([pa for pa in hits if (not low_conf_only) or is_low(pa)],
                              key=lambda pa: 0 if is_low(pa) else 1)
                for pa in cand[: cur - h_target]:
                    msgs.append(f"{name}: DEMOTE {pa.get('result')}@inn{pa.get('inning')}->Out (H={h_target}, had {cur})")
                    pa["result"] = "Out"
                    pa["run_scored"] = False
            else:
                cand = sorted([pa for pa in pas if (pa.get("result") or "").upper() not in HITS
                               and ((not low_conf_only) or is_low(pa))],
                              key=lambda pa: 0 if is_low(pa) else 1)
                need = h_target - cur
                for pa in cand[:need]:
                    msgs.append(f"{name}: PROMOTE {pa.get('result')}@inn{pa.get('inning')}->1B (H={h_target}, had {cur})")
                    pa["result"] = "1B"
                if len(cand) < need:
                    msgs.append(f"{name}: WARN H={h_target} but only {len(cand)} "
                                f"{'low-conf ' if low_conf_only else ''}candidate(s) (had {cur})")
    return msgs


def _hit_type_from_notes(pa: dict) -> str:
    """Infer hit base from step-1 evidence already in the notes: X_COUNT (0->1B,
    1->2B, 2->3B) or crossbar count. Defaults to 1B."""
    notes = pa.get("notes") or ""
    m = re.search(r"X_COUNT\s*=?\s*([0-3])", notes)
    if m:
        return {"0": "1B", "1": "2B", "2": "3B", "3": "3B"}[m.group(1)]
    m = re.search(r"(\d+)\s*crossbar", notes)
    if m:
        return {0: "1B", 1: "1B", 2: "2B", 3: "3B"}.get(int(m.group(1)), "1B")
    return "1B"


def _enforce_hits_double_entry(raw_json: dict, card_hits: list[int],
                               stats_by_slot: dict[int, list[tuple[int, int]]],
                               innings: int) -> list[str]:
    """Assign hit cells using BOTH margins: per-player H (row sums) and per-inning
    hits (column sums). A greedy fill marks the most hit-likely cells first (step-1
    hit evidence > current hit > low-confidence > reached-base > out) while never
    exceeding a player's H or an inning's hit count. This pins WHICH cells are hits;
    the zoomed second pass later reads the TYPE (1B/2B/3B). Mutates raw_json.
    """
    HITS = {"1B", "2B", "3B", "HR"}
    row_need: dict[int, int] = {}
    player_pas: dict[int, list[dict]] = {}
    for slot in raw_json.get("lineup", []):
        targets = stats_by_slot.get(slot.get("batting_order"), [])
        for idx, p in enumerate(slot.get("players", [])):
            if idx >= len(targets):
                continue
            row_need[id(p)] = targets[idx][0]
            player_pas[id(p)] = p.get("plate_appearances", [])
    col_need = {i: (card_hits[i - 1] if i - 1 < len(card_hits) else 0) for i in range(1, innings + 1)}

    def pref(pa: dict) -> int:
        res = (pa.get("result") or "").upper()
        notes = pa.get("notes") or ""
        if res in HITS:
            return 5
        if "HIT_STROKE" in notes or re.search(r"X_COUNT\s*[1-9]", notes):
            return 4
        if pa.get("confidence") == "low":
            return 2
        if res in {"BB", "FC", "HBP"} or (res.startswith("E") and res[1:].isdigit()):
            return 1
        return 0

    cand = []
    for pid, pas in player_pas.items():
        for pa in pas:
            cand.append((pref(pa), pid, int(pa.get("inning", 0)), pa))
    cand.sort(key=lambda c: -c[0])

    rn, cn, assigned = dict(row_need), dict(col_need), set()
    for _, pid, inn, pa in cand:
        if rn.get(pid, 0) > 0 and cn.get(inn, 0) > 0:
            assigned.add(id(pa))
            rn[pid] -= 1
            cn[inn] -= 1

    msgs: list[str] = []
    for pas in player_pas.values():
        for pa in pas:
            is_hit = (pa.get("result") or "").upper() in HITS
            should = id(pa) in assigned
            if should and not is_hit:
                new_type = _hit_type_from_notes(pa)
                msgs.append(f"inn{pa.get('inning')}: {pa.get('result')}->{new_type} (double-entry: row+col say hit)")
                pa["result"] = new_type
            elif is_hit and not should:
                msgs.append(f"inn{pa.get('inning')}: {pa.get('result')}->Out (double-entry: not a hit cell)")
                pa["result"] = "Out"
                pa["run_scored"] = False
    leftover = sum(v for v in rn.values() if v > 0) + sum(v for v in cn.values() if v > 0)
    if leftover:
        msgs.append(f"WARN double-entry left {leftover} hit(s) unplaced (margins infeasible / grid gaps)")
    return msgs


def _enforce_player_walks(raw_json: dict, stats_by_slot: dict[int, list[tuple[int, int]]]) -> list[str]:
    """Each player has exactly PA - AB non-at-bat events (BB/HBP). Among their
    NON-hit PAs (hits already pinned by double-entry), bring the BB/HBP count to
    that target. Run-safe: only promotes outs->BB (adds no run) and only demotes
    NON-scoring walks (removes no run), so per-inning run totals stay intact.
    """
    NON_AB = {"BB", "HBP"}
    HITS = {"1B", "2B", "3B", "HR"}
    msgs: list[str] = []

    def is_low(pa):
        return pa.get("confidence", "high") == "low"

    for slot in raw_json.get("lineup", []):
        targets = stats_by_slot.get(slot.get("batting_order"), [])
        for idx, p in enumerate(slot.get("players", [])):
            if idx >= len(targets):
                continue
            ab = targets[idx][1]
            pas = p.get("plate_appearances", [])
            target = len(pas) - ab
            if target < 0:
                continue
            name = p.get("name", f"slot{slot.get('batting_order')}")
            nonhit = [pa for pa in pas if (pa.get("result") or "").upper() not in HITS]
            walks = [pa for pa in nonhit if (pa.get("result") or "").upper() in NON_AB]
            n = len(walks)
            if n == target:
                continue
            if n > target:
                demotable = sorted([pa for pa in walks if not pa.get("run_scored")],
                                   key=lambda pa: 0 if is_low(pa) else 1)
                for pa in demotable[: n - target]:
                    msgs.append(f"{name}: inn{pa.get('inning')} {pa.get('result')}->Out (walks={target}, had {n})")
                    pa["result"] = "Out"
                if len(demotable) < n - target:
                    msgs.append(f"{name}: WARN walks={target} but {n} present (some scored — left as-is)")
            else:
                cand = sorted([pa for pa in nonhit if (pa.get("result") or "").upper() not in NON_AB],
                              key=lambda pa: 0 if is_low(pa) else 1)
                need = target - n
                for pa in cand[:need]:
                    msgs.append(f"{name}: inn{pa.get('inning')} {pa.get('result')}->BB (walks={target}, had {n})")
                    pa["result"] = "BB"
                if len(cand) < need:
                    msgs.append(f"{name}: WARN walks={target} but only {len(cand)} candidate(s) (had {n})")
    return msgs


def _check_player_ab(raw_json: dict, stats_by_slot: dict[int, list[tuple[int, int]]]) -> list[str]:
    """Advisory: PA - AB = number of non-at-bat events (BB/HBP/SAC/SF). Flag when
    the extracted count of those disagrees (often a missed/extra walk)."""
    NON_AB = {"BB", "HBP", "SAC", "SF"}
    msgs: list[str] = []
    for slot in raw_json.get("lineup", []):
        bo = slot.get("batting_order")
        targets = stats_by_slot.get(bo, [])
        for idx, p in enumerate(slot.get("players", [])):
            if idx >= len(targets):
                continue
            ab = targets[idx][1]
            pas = p.get("plate_appearances", [])
            expected = len(pas) - ab
            got = sum(1 for pa in pas if (pa.get("result") or "").upper() in NON_AB)
            if expected != got:
                msgs.append(f"{p.get('name', f'slot{bo}')}: expected {expected} non-AB (PA{len(pas)}-AB{ab}), "
                            f"found {got} BB/HBP")
    return msgs


def _first_monotonic_run(nums: list[int], length: int) -> list[int] | None:
    """Return the first non-decreasing window of >=length ints (the cumulative
    row's signature, e.g. 3 3 5 5 5 6 9 9 12), else None."""
    for start in range(0, len(nums) - length + 1):
        window = nums[start:start + length]
        if all(window[i + 1] >= window[i] for i in range(len(window) - 1)):
            return window
    return None


def _parse_totals(text: str, innings: int = 9) -> dict | None:
    """Parse the totals-call output. Reads the per-inning RUNS/HITS/ERRORS/LOB
    quadrant rows and the cumulative row. Returns a dict with those lists plus
    'runs_per_inning' (authoritative: cumulative differenced, else the RUNS row),
    or None if no run signal could be recovered.
    """
    def grab(prefix: str) -> list[int] | None:
        for line in text.splitlines():
            u = line.strip().upper()
            if u.startswith(prefix) and "NONE" not in u:
                nums = [int(x) for x in re.findall(r"-?\d+", line.split(":", 1)[1])]
                if len(nums) >= innings:
                    return nums[:innings]
        return None

    runs = grab("RUNS:")
    hits = grab("HITS:")
    errors = grab("ERRORS:")
    lob = grab("LOB:")
    cum = grab("CUMULATIVE:")

    # Chatty fallback: the model often emits per-inning quadrants as prose, e.g.
    # "Inning 1: TL=2, TR=1, BL=0, BR=3". Parse those into the four rows.
    if runs is None:
        blocks = re.split(r"inning\s*(\d+)\s*[:.\-]", text, flags=re.IGNORECASE)
        data: dict[int, tuple[int, int, int, int]] = {}
        for i in range(1, len(blocks) - 1, 2):
            try:
                inn = int(blocks[i])
            except ValueError:
                continue
            blk = blocks[i + 1]

            def q(lbl: str, b: str = blk) -> int:
                m = re.search(lbl + r"\s*=\s*(\d+)", b, re.IGNORECASE)
                return int(m.group(1)) if m else 0

            if 1 <= inn <= innings and re.search(r"[TB][LR]\s*=", blk, re.IGNORECASE):
                data[inn] = (q("TL"), q("TR"), q("BL"), q("BR"))
        if data:
            errors = [data.get(i, (0, 0, 0, 0))[0] for i in range(1, innings + 1)]
            hits = [data.get(i, (0, 0, 0, 0))[1] for i in range(1, innings + 1)]
            lob = [data.get(i, (0, 0, 0, 0))[2] for i in range(1, innings + 1)]
            runs = [data.get(i, (0, 0, 0, 0))[3] for i in range(1, innings + 1)]

    # Fallback for chatty output: a line mentioning "cumulative"/"totaal"
    # (but not "per inning") with a non-decreasing run of >=innings numbers.
    if not cum:
        for line in text.splitlines():
            low = line.lower()
            if ("cumulative" in low or "totaal" in low) and "per inning" not in low:
                run = _first_monotonic_run([int(x) for x in re.findall(r"-?\d+", line)], innings)
                if run:
                    cum = run
                    break
    if not cum:
        cum = _first_monotonic_run([int(x) for x in re.findall(r"-?\d+", text)], innings)

    # Authoritative per-inning runs: prefer the cumulative row differenced
    # (monotonic and easy to read), else the directly-read RUNS row.
    rpi = None
    if cum and len(cum) >= innings:
        cum = cum[:innings]
        diff, prev, ok = [], 0, True
        for v in cum:
            d = v - prev
            if d < 0:
                ok = False
                break
            diff.append(d)
            prev = v
        if ok:
            rpi = diff
    if rpi is None and runs:
        rpi = runs
    if rpi is None:
        return None
    return {
        "runs_per_inning": rpi,
        "runs": runs, "hits": hits, "errors": errors, "lob": lob, "cumulative": cum,
    }


def _enforce_pa_ordering(raw_json: dict) -> list[str]:
    """Deterministically trim phantom PAs so PA counts are non-increasing by
    batting order (the lead-off batter gets the most; each later slot the same
    or fewer). Substitution slots are summed. Removal priority within a slot:
    lowest-confidence first, then highest inning number (most likely a phantom
    from ink bleed or an inning shift). Mutates raw_json; returns log messages.
    """
    msgs: list[str] = []
    conf_rank = {"low": 0, "medium": 1, "high": 2}
    slots = sorted(raw_json.get("lineup", []), key=lambda s: s.get("batting_order", 99))
    prev_cap: int | None = None
    for slot in slots:
        players = slot.get("players", [])
        all_pas = [pa for p in players for pa in p.get("plate_appearances", [])]
        count = len(all_pas)
        if prev_cap is not None and count > prev_cap:
            excess = count - prev_cap
            ranked = sorted(
                all_pas,
                key=lambda pa: (conf_rank.get(pa.get("confidence", "high"), 2),
                                -int(pa.get("inning", 0))),
            )
            remove_ids = {id(pa) for pa in ranked[:excess]}
            removed = []
            for p in players:
                kept = []
                for pa in p.get("plate_appearances", []):
                    if id(pa) in remove_ids:
                        removed.append(f"inn {pa.get('inning')} {pa.get('result')}")
                    else:
                        kept.append(pa)
                p["plate_appearances"] = kept
            msgs.append(
                f"order {slot.get('batting_order')}: trimmed {excess} PA "
                f"({count}->{prev_cap}): " + ", ".join(removed)
            )
            count = prev_cap
        prev_cap = count
    return msgs


_HIT_RESULTS = {"1B", "2B", "3B", "HR"}
_REACHED_RESULTS = {"1B", "2B", "3B", "HR", "BB", "HBP", "FC"}


def _reached_base(result: str | None) -> bool:
    """True if the batter reached base (and could therefore score)."""
    r = (result or "").upper()
    return r in _REACHED_RESULTS or (r.startswith("E") and r[1:].isdigit())


def _reconstruct_grid(card: dict | None, innings: int, num_slots: int = 9) -> dict | None:
    """Derive which batting slot batted in each inning, purely from the card.

    Two facts combine: (1) batters in an inning = 3 + runs + LOB (exact when the
    inning ends on 3 outs — double plays and baserunning outs cancel out), and
    (2) the lineup bats as one continuous cycle 1→2→…→N→1→…, each inning a
    contiguous slice. Slot 1 leads off the game (sequence position 0).

    Returns {bf, total, slot_innings, inning_slots} or None. NOTE: brittle — a
    single wrong per-inning count shifts every later inning's slot assignment,
    and the final inning may be partial (walk-off / home team not batting).
    """
    if not card:
        return None
    runs = card.get("runs_per_inning") or []
    lob = card.get("lob") or []
    bf: list[int] = []
    for i in range(1, innings + 1):
        r = runs[i - 1] if i - 1 < len(runs) else 0
        l = lob[i - 1] if i - 1 < len(lob) else 0
        bf.append(3 + r + l)
    slot_innings: dict[int, list[int]] = {s: [] for s in range(1, num_slots + 1)}
    inning_slots: dict[int, list[int]] = {}
    pos = 0
    for i in range(1, innings + 1):
        slots = []
        for _ in range(bf[i - 1]):
            slot = (pos % num_slots) + 1
            slots.append(slot)
            slot_innings[slot].append(i)
            pos += 1
        inning_slots[i] = slots
    return {"bf": bf, "total": pos, "slot_innings": slot_innings, "inning_slots": inning_slots}


def _check_continuity(raw_json: dict, num_slots: int = 9) -> list[str]:
    """Structural cross-check needing no card data: the lineup bats as one
    continuous cycle 1→2→…→N→1→…, so each inning's batting slots form a
    contiguous arc and inning N+1 starts right after inning N ends. Flags arc
    breaks — a slot that should have batted but didn't (missing PA) or one that
    appears out of sequence (phantom / wrong inning / duplicate). Returns
    messages; empty list means continuity holds.
    """
    inn_slots: dict[int, set[int]] = {}
    for slot in raw_json.get("lineup", []):
        bo = slot.get("batting_order")
        if bo is None:
            continue
        for p in slot.get("players", []):
            for pa in p.get("plate_appearances", []):
                inn_slots.setdefault(int(pa.get("inning", 0)), set()).add(int(bo))

    msgs: list[str] = []
    expected = 1
    for inn in sorted(inn_slots):
        slots = inn_slots[inn]
        arc = [((expected - 1 + i) % num_slots) + 1 for i in range(len(slots))]
        if set(arc) != slots:
            missing = sorted(set(arc) - slots)
            extra = sorted(slots - set(arc))
            detail = []
            if missing:
                detail.append(f"missing slot(s) {missing}")
            if extra:
                detail.append(f"out-of-sequence slot(s) {extra}")
            msgs.append(f"inn {inn}: expected batters {arc}, got {sorted(slots)} — " + "; ".join(detail))
        if arc:
            expected = (arc[-1] % num_slots) + 1
    return msgs


def _realign_innings(raw_json: dict, grid: dict) -> list[str]:
    """Relabel each slot's PA innings to the reconstructed grid. Applied per slot
    ONLY when that slot's extracted PA count equals the reconstructed count (else
    the mapping is ambiguous and the slot is skipped). Mutates raw_json.
    """
    slot_innings = grid["slot_innings"]
    msgs: list[str] = []
    for slot in raw_json.get("lineup", []):
        bo = slot.get("batting_order")
        target = slot_innings.get(bo, [])
        pas = [pa for p in slot.get("players", []) for pa in p.get("plate_appearances", [])]
        if len(pas) != len(target):
            msgs.append(f"order {bo}: SKIP realign (have {len(pas)} PA, grid expects {len(target)})")
            continue
        changes = []
        for pa, new_inn in zip(sorted(pas, key=lambda x: int(x.get("inning", 0))), target):
            old = int(pa.get("inning", 0))
            if old != new_inn:
                changes.append(f"{old}->{new_inn}({pa.get('result')})")
                pa["inning"] = new_inn
        if changes:
            msgs.append(f"order {bo}: realigned " + ", ".join(changes))
    return msgs


def _enforce_batters_faced(raw_json: dict, card: dict | None, innings: int) -> list[str]:
    """Use the card's per-inning totals to clean up per-inning over-attribution.

    Only acts where it is PROVABLY safe: an inning the card shows as fully empty
    (0 runs, 0 hits, 0 errors, 0 LOB) had nobody reach base and at most 3 batters
    (all outs). There we drop any reached-base PA (a walk/hit in such an inning is
    impossible) and cap the inning at 3 outs (lowest-confidence first).

    For non-empty innings the exact bound is batters = 3 + runs + LOB, but the
    LOB read is noisy and removal can't reliably tell a real hit from a phantom
    when an inning is over-attributed, so those are ADVISORY-only (no trimming).
    Mutates raw_json; returns log messages.
    """
    if not card:
        return []
    runs = card.get("runs_per_inning") or []
    hits = card.get("hits") or []
    errors = card.get("errors") or []
    lob = card.get("lob") or []
    msgs: list[str] = []

    by_inn: dict[int, list[dict]] = {}
    for slot in raw_json.get("lineup", []):
        for p in slot.get("players", []):
            for pa in p.get("plate_appearances", []):
                by_inn.setdefault(int(pa.get("inning", 0)), []).append(pa)

    def g(lst: list[int], i: int) -> int:
        return lst[i - 1] if i - 1 < len(lst) else 0

    def is_low(pa: dict) -> bool:
        return pa.get("confidence", "high") == "low"

    remove_ids: set[int] = set()
    for inn in range(1, innings + 1):
        entries = by_inn.get(inn, [])
        if not entries:
            continue
        r, h, e, l = g(runs, inn), g(hits, inn), g(errors, inn), g(lob, inn)
        fully_empty = (r == 0 and h == 0 and e == 0 and l == 0)
        if fully_empty:
            reached = [pa for pa in entries if _reached_base(pa.get("result"))]
            outs = sorted([pa for pa in entries if not _reached_base(pa.get("result"))],
                          key=lambda pa: 0 if is_low(pa) else 1)
            for pa in reached:  # impossible in a 0-reached inning
                remove_ids.add(id(pa))
                msgs.append(f"inn {inn}: drop reached-base PA {pa.get('result')} "
                            f"(card shows inning fully empty)")
            for pa in outs[3:]:  # at most 3 batters in a 1-2-3 inning
                remove_ids.add(id(pa))
                msgs.append(f"inn {inn}: drop out PA {pa.get('result')} (>3 in an empty inning)")
        else:
            cap = 3 + r + l
            if len(entries) > cap:
                msgs.append(f"inn {inn}: ADVISORY {len(entries)} PAs > ~{cap} expected "
                            f"(3+runs{r}+lob{l}) — likely over-attributed (no auto-trim)")

    if remove_ids:
        for slot in raw_json.get("lineup", []):
            for p in slot.get("players", []):
                p["plate_appearances"] = [
                    pa for pa in p.get("plate_appearances", []) if id(pa) not in remove_ids
                ]
    return msgs


def _enforce_inning_totals(raw_json: dict, card: dict | None, innings: int,
                           hits_low_conf_only: bool = True, do_hits: bool = True) -> list[str]:
    """Force per-inning RUNS and HITS to match the card's 'Totaal per inning'.

    RUNS (double-confirmed on the card → trusted): flip run_scored among the
    inning's reached-base PAs until the count matches; prefer low-confidence PAs.
    HITS (single read → conservative): only promote/demote LOW-confidence PAs by
    default; if an inning lacks a low-confidence candidate, leave it and warn.
    Mutates raw_json; returns log messages. Errors/LOB are intentionally ignored.
    """
    if not card:
        return []
    msgs: list[str] = []
    card_runs = card.get("runs_per_inning")
    card_hits = card.get("hits")

    by_inn: dict[int, list[dict]] = {}
    for slot in raw_json.get("lineup", []):
        for p in slot.get("players", []):
            for pa in p.get("plate_appearances", []):
                by_inn.setdefault(int(pa.get("inning", 0)), []).append(pa)

    def is_low(pa: dict) -> bool:
        return pa.get("confidence", "high") == "low"

    # ── RUNS ────────────────────────────────────────────────────────────────
    if card_runs:
        for inn in range(1, innings + 1):
            if inn - 1 >= len(card_runs):
                continue
            target = card_runs[inn - 1]
            pas = by_inn.get(inn, [])
            scored = [pa for pa in pas if pa.get("run_scored")]
            cur = len(scored)
            if cur > target:  # phantom runs → turn lowest-confidence off
                for pa in sorted(scored, key=lambda x: 0 if is_low(x) else 1)[: cur - target]:
                    pa["run_scored"] = False
                    msgs.append(f"inn {inn}: run OFF on {pa.get('result')} (card={target}, had={cur})")
            elif cur < target:  # missed runs → turn eligible PAs on
                elig = [pa for pa in pas if _reached_base(pa.get("result")) and not pa.get("run_scored")]
                elig.sort(key=lambda x: 0 if is_low(x) else 1)  # low-confidence first
                need = target - cur
                for pa in elig[:need]:
                    pa["run_scored"] = True
                    msgs.append(f"inn {inn}: run ON on {pa.get('result')} (card={target}, had={cur})")
                if len(elig) < need:
                    msgs.append(f"inn {inn}: WARN card={target} runs but only {len(elig)} eligible PAs")

    # ── HITS (conservative) — skipped when per-player H is available ──────────
    if card_hits and do_hits:
        for inn in range(1, innings + 1):
            if inn - 1 >= len(card_hits):
                continue
            target = card_hits[inn - 1]
            pas = by_inn.get(inn, [])
            hits = [pa for pa in pas if (pa.get("result") or "").upper() in _HIT_RESULTS]
            cur = len(hits)
            if cur == target:
                continue
            if cur > target:  # demote extra hits (lowest-confidence) to an out
                cand = [pa for pa in hits if (not hits_low_conf_only) or is_low(pa)]
                cand.sort(key=lambda x: 0 if is_low(x) else 1)
                for pa in cand[: cur - target]:
                    old = pa.get("result")
                    pa["result"] = "Out"
                    pa["run_scored"] = False
                    msgs.append(f"inn {inn}: hit DEMOTE {old}->Out (card={target}, had={cur})")
            else:  # promote non-hit PAs to 1B
                nonhit = [pa for pa in pas if (pa.get("result") or "").upper() not in _HIT_RESULTS]
                cand = [pa for pa in nonhit if (not hits_low_conf_only) or is_low(pa)]
                cand.sort(key=lambda x: 0 if is_low(x) else 1)
                need = target - cur
                for pa in cand[:need]:
                    old = pa.get("result")
                    pa["result"] = "1B"
                    msgs.append(f"inn {inn}: hit PROMOTE {old}->1B (card={target}, had={cur})")
                if len(cand) < need:
                    msgs.append(f"inn {inn}: WARN card={target} hits but only {len(cand)} "
                                f"{'low-conf ' if hits_low_conf_only else ''}candidates (had {cur})")
    return msgs


@click.command()
@click.argument("image_or_dir", type=click.Path(exists=True))

@click.option("--provider", default="anthropic", show_default=True,
              type=click.Choice(["anthropic", "google"]),
              help="LLM provider to use")
@click.option("--model", default=None,
              help="Model name (default: claude-opus-4-5 for anthropic, gemini-2.0-flash for google)")
@click.option("--players-file", default=None, type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True)
@click.option("--out", default=None, help="Path to save the archive JSON")
@click.option("--date", "date_str", default=None, help="Game date YYYY-MM-DD")
@click.option("--opponent", default=None, help="Away team name")
@click.option("--innings", default=9, show_default=True, help="Number of innings played")
@click.option("--players", default=9, show_default=True,
              help="Number of batting-order player rows (bottom total rows beyond this are skipped)")
@click.option("--reuse-step1", is_flag=True, default=False,
              help="Reuse cached step-1 descriptions (skip the per-row vision call) when available. "
                   "Use when iterating on step-2/enforcement without changing the step-1 approach.")
@click.option("--realign", is_flag=True, default=False,
              help="Realign PA innings to the reconstructed batting grid (card per-inning counts + "
                   "lineup continuity). Only applies when grid and extracted PA totals agree. Experimental.")
@click.option("--fresh-totals", is_flag=True, default=False,
              help="Re-read the bottom totals strip via the API and overwrite the cached/edited "
                   "totals table. Default: reuse the hand-editable totals_cache file if it exists.")
@click.option("--stats-image", default=None, type=click.Path(exists=True),
              help="Image containing the far-right per-player H-AB totals column. Defaults to the "
                   "main image (assumes stats column is included there).")
@click.option("--no-stats", is_flag=True, default=False,
              help="Skip per-player H-AB extraction entirely (for games without a stats column).")
@click.option("--fresh-stats", is_flag=True, default=False,
              help="Re-read the per-player H-AB column via the API and overwrite the cached file.")
@click.option("--second-pass", is_flag=True, default=False,
              help="After enforcement, re-read rows containing low-confidence cells on their upscaled "
                   "crops, with full per-player/per-inning constraints, to correct result labels.")
@click.option("--red-lines", "use_red_lines", is_flag=True, default=True, show_default=True,
              help="Use red lines on image for row splitting (ignored if a directory is given)")
@click.option("--export", "do_export", is_flag=True, help="Run Excel export after importing to DB")
@click.option("--export-out", default="stats.xlsx", show_default=True, help="Output path for Excel export")
@click.option("--enhance", "enhance_mode",
              type=click.Choice(["none", "clahe", "esrgan", "both",
                                 "nodiag", "nodiag+esrgan"]),
              default="none", show_default=True,
              help="Pre-process row images before the VLM. "
                   "nodiag=Hough diagonal suppression, "
                   "nodiag+esrgan=suppress then AI-sharpen.")
def main(image_or_dir: str, provider: str, model: str | None, players_file: str | None,
         dry_run: bool, out: str | None, date_str: str | None, opponent: str | None,
         innings: int, players: int, reuse_step1: bool, realign: bool, fresh_totals: bool,
         stats_image: str | None, no_stats: bool, fresh_stats: bool, second_pass: bool,
         use_red_lines: bool, do_export: bool, export_out: str, enhance_mode: str) -> None:
    # Apply default model per provider (env var EXTRACTION_MODEL overrides hardcoded default)
    if model is None:
        if provider == "google":
            model = os.environ.get("EXTRACTION_MODEL_GOOGLE", "gemini-2.5-flash")
        else:
            model = os.environ.get("EXTRACTION_MODEL", "claude-opus-4-5")
    click.echo(f"Provider: {provider}  Model: {model}")
    input_path = Path(image_or_dir)

    # ── Auto-detect date + opponent from filename (YYYY-MM-DD_opponent.ext) ──
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", input_path.stem)
    if m:
        if date_str is None:
            date_str = m.group(1)
        if opponent is None:
            opponent = m.group(2).replace("_", " ").title()
    if date_str:
        click.echo(f"Date: {date_str}  Opponent: {opponent or 'unknown'}")

    # ── Path layout ──────────────────────────────────────────────────────────
    # images/
    #   scans/          ← source images (input lives here)
    #   ground_truth/   ← hand-editable totals + stats txt files
    #   _cache/         ← fully generated; never edited; gitignored
    #     rows/
    #     rows_enhanced_esrgan/
    #     step1/
    #     stats_crop/
    if input_path.is_file():
        images_root = input_path.parent.parent  # images/ (parent of scans/)
        rows_path = images_root / "_cache" / "rows"
        click.echo(f"Splitting {input_path.name} into row crops → {rows_path}")
        if use_red_lines:
            split_paths = split_from_red_lines(str(input_path), rows_path)
        else:
            split_paths = split_scorecard(str(input_path), rows_path)
        if not split_paths:
            click.echo("No rows detected — check the image or try without --red-lines.")
            raise SystemExit(1)
        click.echo(f"Split into {len(split_paths)} row crops.")
    else:
        images_root = input_path.parent
        rows_path = input_path

    cache_root = images_root / "_cache"
    ground_truth_dir = images_root / "ground_truth"

    row_files = sorted(rows_path.glob("*_row*.png"))
    if not row_files:
        row_files = sorted(rows_path.glob("*_row*.jpg"))
    if not row_files:
        click.echo(f"No row images found in {rows_dir}")
        raise SystemExit(1)

    click.echo(f"Found {len(row_files)} row images")

    # Only the batting-order player rows go to step-1. Any extra crops below them
    # (the '10' template row, 'Totaal per inning', 'Totaal') are the bottom total
    # rows — they are read separately via the dedicated totals strip, so skip them
    # here to avoid wasted API calls and polluting step-2 with non-player rows.
    if len(row_files) > players:
        skipped = [p.name for p in row_files[players:]]
        click.echo(f"Skipping {len(skipped)} non-player bottom row(s): {', '.join(skipped)}")
        row_files = row_files[:players]

    totals_candidates = sorted(rows_path.glob("*_totals.png"))
    totals_img = totals_candidates[0] if totals_candidates else None
    if totals_img:
        click.echo(f"Totals strip: {totals_img.name}")
    else:
        click.echo("No totals strip found (pass an image input to auto-generate one).")

    # Cost info for paid providers
    if provider == "google":
        est_calls = len(row_files) + 1
        click.echo(f"Will make {est_calls} API calls to {provider}/{model}. Estimated cost: <$0.05.")

    # Build game context block (injected into both step-1 and step-2)
    game_context_lines = [
        f"This game lasted {innings} innings.",
        f"Inning columns {innings + 1} through 12 in the header are EMPTY template columns — do NOT describe their contents as plate appearances.",
        "The far-right area of the scorecard contains STATISTICS boxes (PA, AB, H, AVG, etc.) — these are totals, NOT plate appearances.",
    ]
    if date_str:
        game_context_lines.insert(0, f"Game date: {date_str}")
    if opponent:
        game_context_lines.insert(0, f"Away team (opponent): {opponent}")
    game_context = "\n\nGAME CONTEXT:\n" + "\n".join(f"  - {l}" for l in game_context_lines)

    # Build roster hint
    roster_hint = ""
    if players_file:
        names = []
        for line in open(players_file, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(",")[0].strip()
            if name:
                names.append(name)
        roster_hint = "\n\nKNOWN ROSTER (Quick): " + ", ".join(names)
        click.echo(f"Loaded {len(names)} player names")

    # ── Build provider clients ────────────────────────────────────────────
    step1_user_text = (
        f"This image shows ONE player row from a KNBSB scorecard. "
        f"The top strip is the header (inning numbers). "
        f"The bottom strip is a single player's scoring cells. "
        f"Red lines mark the exact boundaries of each sub-cell — "
        f"use them to identify which sub-cell each marking is in. "
        f"This game lasted {innings} innings — only describe PA cells "
        f"for innings 1 through {innings}. "
        f"Columns {innings + 1}+ and the stat boxes on the right are NOT scoring cells. "
        f"Describe the player info and every PA cell as instructed."
    )
    step1_system = _STEP1_SYSTEM + roster_hint + game_context

    def _google_client():
        from google import genai as gai
        key = os.environ.get("GOOGLE_API_KEY", "")
        # AIzaSy* keys: plain API key — pass directly
        if key.startswith("AIzaSy"):
            return gai.Client(api_key=key)
        # AQ.* keys don't work with the REST endpoint; fall through to ADC.
        # ADC is set up via: gcloud auth application-default login
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return gai.Client(credentials=creds)

    def call_step1(row_path: Path) -> str:
        if provider == "google":
            import time
            from google.genai import types as gtypes
            from google.genai.errors import ServerError
            gclient = _google_client()
            img_bytes = row_path.read_bytes()
            for attempt in range(4):
                try:
                    response = gclient.models.generate_content(
                        model=model,
                        contents=[
                            gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                            step1_user_text,
                        ],
                        config=gtypes.GenerateContentConfig(
                            system_instruction=step1_system,
                            max_output_tokens=4096,
                        ),
                    )
                    return response.text
                except ServerError as e:
                    if attempt < 3 and "503" in str(e):
                        wait = 15 * (attempt + 1)
                        click.echo(f"    503 overloaded — retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
        else:
            aclient = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            img_data = base64.standard_b64encode(row_path.read_bytes()).decode()
            msg = aclient.messages.create(
                model=model,
                max_tokens=4096,
                system=step1_system,
                messages=[{"role": "user", "content": [
                    _make_image_content(img_data, "image/png"),
                    {"type": "text", "text": step1_user_text},
                ]}],
            )
            return msg.content[0].text

    def call_step2(prompt: str) -> str:
        if provider == "google":
            import time
            from google.genai import types as gtypes
            from google.genai.errors import ServerError
            gclient = _google_client()
            for attempt in range(4):
                try:
                    response = gclient.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=gtypes.GenerateContentConfig(
                            system_instruction=_STEP2_SYSTEM,
                            temperature=0,
                            max_output_tokens=32768,
                        ),
                    )
                    return response.text
                except ServerError as e:
                    if attempt < 3 and "503" in str(e):
                        wait = 15 * (attempt + 1)
                        click.echo(f"    503 overloaded — retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
        else:
            aclient = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            msg = aclient.messages.create(
                model=model,
                max_tokens=8192,
                temperature=0,
                system=_STEP2_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

    def call_totals(path: Path) -> str:
        if provider == "google":
            from google.genai import types as gtypes
            gclient = _google_client()
            response = gclient.models.generate_content(
                model=model,
                contents=[
                    gtypes.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"),
                    "Read the totals rows.",
                ],
                config=gtypes.GenerateContentConfig(
                    system_instruction=_TOTALS_SYSTEM, temperature=0, max_output_tokens=1024,
                ),
            )
            return response.text
        aclient = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        img_data = base64.standard_b64encode(path.read_bytes()).decode()
        msg = aclient.messages.create(
            model=model, max_tokens=1024, temperature=0, system=_TOTALS_SYSTEM,
            messages=[{"role": "user", "content": [
                _make_image_content(img_data, "image/png"),
                {"type": "text", "text": "Read the totals rows."},
            ]}],
        )
        return msg.content[0].text

    def call_stats(path: Path) -> str:
        if provider == "google":
            from google.genai import types as gtypes
            gclient = _google_client()
            response = gclient.models.generate_content(
                model=model,
                contents=[
                    gtypes.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"),
                    "Read the per-player H-AB totals.",
                ],
                config=gtypes.GenerateContentConfig(
                    system_instruction=_STATS_SYSTEM, temperature=0, max_output_tokens=512,
                ),
            )
            return response.text
        aclient = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        img_data = base64.standard_b64encode(path.read_bytes()).decode()
        msg = aclient.messages.create(
            model=model, max_tokens=512, temperature=0, system=_STATS_SYSTEM,
            messages=[{"role": "user", "content": [
                _make_image_content(img_data, "image/png"),
                {"type": "text", "text": "Read the per-player H-AB totals."},
            ]}],
        )
        return msg.content[0].text

    def call_refine(path: Path, system: str) -> str:
        if provider == "google":
            from google.genai import types as gtypes
            gclient = _google_client()
            response = gclient.models.generate_content(
                model=model,
                contents=[
                    gtypes.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"),
                    "Re-read this player's innings as instructed.",
                ],
                config=gtypes.GenerateContentConfig(
                    system_instruction=system, temperature=0, max_output_tokens=512,
                ),
            )
            return response.text
        aclient = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        img_data = base64.standard_b64encode(path.read_bytes()).decode()
        msg = aclient.messages.create(
            model=model, max_tokens=512, temperature=0, system=system,
            messages=[{"role": "user", "content": [
                _make_image_content(img_data, "image/png"),
                {"type": "text", "text": "Re-read this player's innings as instructed."},
            ]}],
        )
        return msg.content[0].text

    # ── Optional image enhancement ───────────────────────────────────────
    if enhance_mode != "none":
        import cv2 as _cv2
        from enhance import enhance_image, ensure_model, DEFAULT_MODEL_PATH
        if enhance_mode in ("esrgan", "both", "nodiag+esrgan"):
            ensure_model(DEFAULT_MODEL_PATH)
        # Cache enhanced crops per mode so repeated step-1/step-2 iteration does
        # not re-run the (slow) ESRGAN pass. A cached crop is reused when it
        # exists and is newer than its source row image.
        enhanced_dir = cache_root / f"rows_enhanced_{enhance_mode}"
        enhanced_dir.mkdir(exist_ok=True)
        enhanced_files: list[Path] = []
        click.echo(f"Enhancing {len(row_files)} rows with mode={enhance_mode} (cache: {enhanced_dir.name})...")
        cached = made = 0
        for i, rp in enumerate(row_files, 1):
            out_path = enhanced_dir / rp.name
            if out_path.exists() and out_path.stat().st_mtime >= rp.stat().st_mtime:
                enhanced_files.append(out_path)
                cached += 1
                click.echo(f"  Row {i}/{len(row_files)}: cached")
                continue
            img = _cv2.imread(str(rp))
            out_img = enhance_image(img, enhance_mode)
            _cv2.imwrite(str(out_path), out_img)
            enhanced_files.append(out_path)
            made += 1
            click.echo(f"  Row {i}/{len(row_files)}: enhanced")
        row_files = enhanced_files
        click.echo(f"Enhanced crops in {enhanced_dir} ({made} new, {cached} cached)")

    # ── Step 1: visual scan per row (cached) ─────────────────────────────
    # Descriptions are cached per row + enhance-mode. With --reuse-step1 a cached
    # description is replayed instead of calling the vision model — letting you
    # iterate on step-2/enforcement at zero step-1 cost. Fresh calls always
    # (re)write the cache.
    step1_cache_dir = cache_root / "step1"
    step1_cache_dir.mkdir(exist_ok=True)
    descriptions: list[str] = []
    reused = 0
    for i, row_path in enumerate(row_files, 1):
        cache_file = step1_cache_dir / f"{row_path.stem}__{enhance_mode}.txt"
        if reuse_step1 and cache_file.exists():
            desc = cache_file.read_text(encoding="utf-8")
            reused += 1
            click.echo(f"  Step 1  row {i}/{len(row_files)}: {row_path.name}  (cached, {len(desc)} chars)")
        else:
            click.echo(f"  Step 1  row {i}/{len(row_files)}: {row_path.name}")
            desc = call_step1(row_path)
            cache_file.write_text(desc, encoding="utf-8")
            click.echo(f"    {len(desc)} chars")
        descriptions.append(f"--- PLAYER ROW {i} ---\n{desc}")
    if reused:
        click.echo(f"Reused {reused}/{len(row_files)} cached step-1 descriptions (no API call)")

    combined = "\n\n".join(descriptions)
    click.echo(f"\nCombined description: {len(combined)} chars. Running step 2...")

    # Build step-2 preamble with game metadata so the parser can fill game.date / game.teams
    step2_preamble_lines = [
        "Parse the descriptions below into JSON.",
        f"This game lasted {innings} innings. DISCARD any PA assigned to inning > {innings} — those are stat columns.",
    ]
    if date_str:
        step2_preamble_lines.append(f"Game date: {date_str}  (use this as game.date in the JSON)")
    if opponent:
        step2_preamble_lines.append(f"Away team: {opponent}  (use this as game.teams.away in the JSON)")
    step2_preamble = "\n".join(step2_preamble_lines) + "\n\n---\n\n"

    # ── Step 2: parse combined description into JSON ──────────────────────
    raw_text = call_step2(step2_preamble + combined)

    # Strip optional markdown fences (handles ```json ... ``` or ``` ... ```)
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped).strip()
        raw_text = stripped

    try:
        raw_json = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Dump raw output so we can inspect what the model returned
        debug_path = rows_path.parent / "step2_debug.txt"
        debug_path.write_text(raw_text, encoding="utf-8")
        click.echo(f"\nJSON parse error: {e}")
        click.echo(f"Raw step-2 output saved to: {debug_path}")
        click.echo("--- first 600 chars ---")
        click.echo(raw_text[:600])
        raise SystemExit(1)

    # ── Read the bottom totals strip (card ground truth) FIRST, then enforce in
    #    order: (1) trim per-inning phantom PAs via the batters-faced bound,
    #    (2) PA-ordering, (3) per-inning RUNS+HITS. All before archive/validation.
    card: dict | None = None
    if totals_img is not None:
        ground_truth_dir.mkdir(exist_ok=True)
        totals_cache_file = ground_truth_dir / f"{input_path.stem}_totals.txt"
        if totals_cache_file.exists() and not fresh_totals:
            card = _read_totals_cache(totals_cache_file, innings)
            click.echo(f"\nUsing hand-verified totals from {totals_cache_file.name} "
                       f"(use --fresh-totals to re-read from the image)")
        else:
            try:
                totals_raw = call_totals(totals_img)
                click.echo(f"\nTotals strip read: {totals_raw.strip().replace(chr(10), ' | ')}")
                card = _parse_totals(totals_raw, innings)
            except Exception as e:  # noqa: BLE001
                click.echo(f"Totals call failed: {e}")
            if card:
                _write_totals_cache(totals_cache_file, card, innings)
                click.echo(f"Wrote totals table to {totals_cache_file} — "
                           f"open it, correct any numbers, then re-run to lock them in.")
    # ── Per-player H-AB totals ───────────────────────────────────────────────
    player_stats: dict[int, list[tuple[int, int]]] | None = None
    if not no_stats:
        stats_src = Path(stats_image) if stats_image is not None else input_path
        ground_truth_dir.mkdir(exist_ok=True)
        stats_cache_file = ground_truth_dir / f"{input_path.stem}_stats.txt"
        if stats_cache_file.exists() and not fresh_stats:
            player_stats = _read_player_stats_cache(stats_cache_file)
            click.echo(f"\nUsing hand-verified player stats from {stats_cache_file.name} "
                       f"(use --fresh-stats to re-read)")
        else:
            try:
                stats_crop = _crop_stats_strip(stats_src, cache_root / "stats_crop")
                rows = _parse_player_stats(call_stats(stats_crop))
                _write_player_stats_cache(stats_cache_file, rows)
                player_stats = _read_player_stats_cache(stats_cache_file)
                click.echo(f"\nExtracted per-player H-AB to {stats_cache_file} — review/edit, then re-run.")
            except Exception as e:  # noqa: BLE001
                click.echo(f"Player-stats extraction failed: {e}")
    if player_stats:
        click.echo("Per-player H by slot: "
                   + ", ".join(f"{s}:{'/'.join(str(h) for h, _ in player_stats[s])}"
                               for s in sorted(player_stats)))

    if card:
        click.echo(f"Card per-inning  runs={card.get('runs_per_inning')}  hits={card.get('hits')}  "
                   f"errors={card.get('errors')}  lob={card.get('lob')}")
        bf_msgs = _enforce_batters_faced(raw_json, card, innings)
        if bf_msgs:
            click.echo("\n-- Batters-faced: trimmed per-inning phantom PAs --")
            for m in bf_msgs:
                click.echo(f"  {m}")

    order_msgs = _enforce_pa_ordering(raw_json)
    if order_msgs:
        click.echo("\n-- PA ordering: trimmed phantom PAs --")
        for m in order_msgs:
            click.echo(f"  {m}")

    # ── Batting-grid reconstruction (diagnostic; opt-in realignment) ─────────
    grid = _reconstruct_grid(card, innings) if card else None
    if grid:
        extracted_total = sum(len(p.get("plate_appearances", []))
                              for s in raw_json.get("lineup", []) for p in s.get("players", []))
        recon = grid["slot_innings"]
        click.echo("\n-- Batting-grid reconstruction (card per-inning counts + lineup continuity) --")
        click.echo(f"  batters/inning: {grid['bf']}  grid total={grid['total']}  extracted PAs={extracted_total}")
        click.echo(f"  reconstructed per-slot PA: {[len(recon[s]) for s in sorted(recon)]}")
        for s in sorted(recon):
            click.echo(f"    slot {s}: innings {recon[s]}")
        if realign:
            # Per-slot realignment: _realign_innings only relabels slots whose PA
            # count matches the grid, so it is safe to run even when the global
            # total disagrees — correct slots get fixed; mismatched ones are skipped
            # and surface as second-pass candidates.
            rmsgs = _realign_innings(raw_json, grid)
            if rmsgs:
                click.echo("  -- realigned PA innings to grid (per-slot; mismatched slots skipped) --")
                for m in rmsgs:
                    click.echo(f"    {m}")
        if grid["total"] != extracted_total:
            click.echo(f"  NOTE: grid total {grid['total']} != extracted {extracted_total} — "
                       f"some slots have missing/extra PAs (second-pass candidates)")

    # ── Second pass: re-read rows with low-confidence cells, with constraints ─
    if second_pass:
        click.echo("\n-- Second pass: re-reading low-confidence rows (constrained) --")
        slot_to_crop = {bo: row_files[bo - 1] for bo in range(1, len(row_files) + 1)}
        any_refined = False
        for slot in sorted(raw_json.get("lineup", []), key=lambda s: s.get("batting_order", 99)):
            bo = slot.get("batting_order")
            slot_players = slot.get("players", [])
            all_pas = [pa for p in slot_players for pa in p.get("plate_appearances", [])]
            if not all_pas or not any(pa.get("confidence") == "low" for pa in all_pas):
                continue
            crop = slot_to_crop.get(bo)
            if crop is None:
                continue
            ctx = [f"- This batting slot batted in innings: {sorted(int(pa.get('inning', 0)) for pa in all_pas)}."]
            targets = player_stats.get(bo, []) if player_stats else []
            for idx, p in enumerate(slot_players):
                pinn = sorted(int(pa.get("inning", 0)) for pa in p.get("plate_appearances", []))
                if idx < len(targets):
                    h, ab = targets[idx]
                    ctx.append(f"- {p.get('name', 'player')} (innings {pinn}): {h} hit(s), "
                               f"{max(len(p.get('plate_appearances', [])) - ab, 0)} walk(s)/HBP.")
            if card:
                ch, ce, cr = card.get("hits") or [], card.get("errors") or [], card.get("runs_per_inning") or []
                for i in sorted(int(pa.get("inning", 0)) for pa in all_pas):
                    parts = []
                    if i - 1 < len(cr):
                        parts.append(f"{cr[i-1]} run(s)")
                    if i - 1 < len(ch):
                        parts.append(f"{ch[i-1]} hit(s)")
                    if i - 1 < len(ce):
                        parts.append(f"{ce[i-1]} error(s)")
                    if parts:
                        ctx.append(f"- Inning {i} (whole team): " + ", ".join(parts) + ".")
            try:
                text = call_refine(crop, _SECOND_PASS_SYSTEM.format(constraints="\n".join(ctx)))
            except Exception as e:  # noqa: BLE001
                click.echo(f"  slot {bo}: refine failed: {e}")
                continue
            changes = []
            for p in slot_players:
                changes += _apply_refinement(p, text)
            names = "/".join(p.get("name", "") for p in slot_players)
            click.echo(f"  slot {bo} ({names}): " + ("; ".join(changes) if changes else "no changes"))
            any_refined = True
        if not any_refined:
            click.echo("  (no low-confidence rows to refine)")

    if card:
        # Per-player H (if available) is more precise than per-inning hits, so let
        # it own hit enforcement; the per-inning pass then only does runs.
        enforce_msgs = _enforce_inning_totals(raw_json, card, innings, do_hits=(player_stats is None))
        if enforce_msgs:
            label = "runs" if player_stats else "runs + hits"
            click.echo(f"\n-- Enforcing card totals ({label}) --")
            for m in enforce_msgs:
                click.echo(f"  {m}")

    if player_stats:
        if card and card.get("hits"):
            de_msgs = _enforce_hits_double_entry(raw_json, card["hits"], player_stats, innings)
            if de_msgs:
                click.echo("\n-- Double-entry hit assignment (per-player H x per-inning hits) --")
                for m in de_msgs:
                    click.echo(f"  {m}")
        else:
            ph_msgs = _enforce_player_hits(raw_json, player_stats)
            if ph_msgs:
                click.echo("\n-- Enforcing per-player hits (from H, no per-inning data) --")
                for m in ph_msgs:
                    click.echo(f"  {m}")
        w_msgs = _enforce_player_walks(raw_json, player_stats)
        if w_msgs:
            click.echo("\n-- Enforcing per-player walks (PA - AB) --")
            for m in w_msgs:
                click.echo(f"  {m}")
        ab_msgs = _check_player_ab(raw_json, player_stats)
        if ab_msgs:
            click.echo("\n-- Per-player AB check (advisory: PA-AB = expected BB/HBP) --")
            for m in ab_msgs:
                click.echo(f"  {m}")

    # ── Lineup-continuity check (structural; needs no card data) ─────────────
    cont_msgs = _check_continuity(raw_json, players)
    click.echo("\n-- Lineup continuity check --")
    if cont_msgs:
        for m in cont_msgs:
            click.echo(f"  *** {m}")
    else:
        click.echo("  OK — batting slots form a continuous cycle across innings")

    # ── Archive ────────────────────────────────────────────────────────────
    if out is None:
        game_date = raw_json.get("game", {}).get("date") or date_str or "unknown"
        opp_slug = (opponent or raw_json.get("game", {}).get("teams", {}).get("away") or "unknown").lower()
        opp_slug = re.sub(r"\s+", "_", opp_slug)
        out_path = Path("data") / "raw" / f"{game_date}_{opp_slug}_rows.json"
    else:
        out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"_step1_description": combined, **raw_json}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(f"Archived to {out_path}")

    # ── Validate + print ───────────────────────────────────────────────────
    _remap_summary_keys(raw_json)
    if "pitching" in raw_json:
        raw_json["pitching"] = [p for p in raw_json["pitching"] if p.get("name")]

    extraction = GameExtraction.model_validate(raw_json)

    game = extraction.game
    click.echo(f"\nGame: {game.teams.get('away','?')} @ {game.teams.get('home','?')}  {game.date}")

    total_runs = 0
    click.echo("\n-- Players -----------------------------------------------")
    for slot in extraction.lineup:
        for player in slot.players:
            pas = player.plate_appearances
            runs = sum(1 for pa in pas if pa.run_scored)
            low = sum(1 for pa in pas if pa.confidence == "low")
            total_runs += runs
            click.echo(f"  [{slot.batting_order}] {player.name}: {len(pas)} PA, {runs} R"
                       + (f"  ({low} low-conf)" if low else ""))
            for pa in pas:
                flag = " [LOW]" if pa.confidence == "low" else ""
                click.echo(f"       Inn {pa.inning}: {pa.result:<6} R={int(pa.run_scored)}{flag}"
                           f"  {pa.notes[:70]}")

    click.echo(f"\nTotal runs attributed: {total_runs}")

    # Per-inning stats we attributed, for cross-checking against the card totals.
    extracted_rpi: dict[int, int] = {}   # runs
    extracted_hpi: dict[int, int] = {}   # hits (1B/2B/3B/HR)
    extracted_epi: dict[int, int] = {}   # reached-on-error (E#)
    _hit_results = {"1B", "2B", "3B", "HR"}
    for slot in extraction.lineup:
        for player in slot.players:
            for pa in player.plate_appearances:
                if pa.run_scored:
                    extracted_rpi[pa.inning] = extracted_rpi.get(pa.inning, 0) + 1
                res = (pa.result or "").upper()
                if res in _hit_results:
                    extracted_hpi[pa.inning] = extracted_hpi.get(pa.inning, 0) + 1
                elif res.startswith("E") and res[1:].isdigit():
                    extracted_epi[pa.inning] = extracted_epi.get(pa.inning, 0) + 1

    # Reconcile against the card totals fetched earlier (post-enforcement these
    # should now match; any residual mismatch is something enforcement could not
    # safely fix — e.g. an inning with no eligible/low-confidence candidate).
    card_rpi = card["runs_per_inning"] if card else None

    def _reconcile(label: str, card_vals: list[int] | None, attributed: dict[int, int],
                   hint_lo: str, hint_hi: str) -> None:
        if not card_vals:
            return
        click.echo(f"\n-- {label} reconciliation (card vs attributed) --")
        n = max(len(card_vals), (max(attributed) if attributed else 0))
        ok = True
        for inn in range(1, n + 1):
            c = card_vals[inn - 1] if inn - 1 < len(card_vals) else 0
            g = attributed.get(inn, 0)
            if c != g:
                ok = False
                click.echo(f"  Inn {inn}: card={c}, attributed={g}  *** {hint_lo if c > g else hint_hi}")
        click.echo(f"  {'OK — matches card' if ok else f'team {label.lower()}: card={sum(card_vals)}, attributed={sum(attributed.values())}'}")

    if card_rpi:
        click.echo(f"\nPer-inning from card  runs={card_rpi}"
                   + (f" hits={card['hits']}" if card and card.get("hits") else "")
                   + (f" errors={card['errors']}" if card and card.get("errors") else "")
                   + (f"  (runs sum={sum(card_rpi)})"))
        _reconcile("Run", card_rpi, extracted_rpi,
                   "MISSED run(s) — check BOT-LEFT/center dot", "PHANTOM run(s) — check for false run marks")
        if card and card.get("hits"):
            _reconcile("Hit", card["hits"], extracted_hpi,
                       "MISSED hit(s) — a hit was read as an out/other", "EXTRA hit(s) — an out was read as a hit")
        if card and card.get("errors"):
            _reconcile("Error", card["errors"], extracted_epi,
                       "MISSED E# — an error-reach read as something else", "EXTRA E# — check those results")
    else:
        click.echo("\n(no card totals available for reconciliation)")

    # ── PA ordering validation ─────────────────────────────────────────────
    # Baseball rule: lead-off batter gets the most PAs; each successive batter
    # gets the same or fewer.  Violations almost always mean phantom PAs.
    click.echo("\n-- PA count check ----------------------------------------")
    pa_by_order: dict[int, tuple[str, int]] = {}
    for slot in extraction.lineup:
        # For substitution slots (multiple players), sum their PAs together
        total_pas = sum(len(p.plate_appearances) for p in slot.players)
        names = "/".join(p.name for p in slot.players)
        pa_by_order[slot.batting_order] = (names, total_pas)

    prev_order, prev_count = None, None
    all_ok = True
    for order in sorted(pa_by_order):
        name, count = pa_by_order[order]
        if prev_count is not None and count > prev_count:
            flag = f"  *** VIOLATION: {count} > {prev_count} (order {prev_order})"
            all_ok = False
        else:
            flag = ""
        click.echo(f"  [{order}] {name}: {count} PA{flag}")
        prev_order, prev_count = order, count
    if all_ok:
        click.echo("  OK — batting order PA counts are non-increasing")

    if dry_run:
        click.echo("\n[dry-run] Nothing written to DB.")
        return

    # ── Write to DB ────────────────────────────────────────────────────────
    db_path = Path("data/season.db")
    init_db(db_path)
    conn = get_connection(db_path)

    existing_id = find_duplicate_game(
        conn,
        date_str or raw_json.get("game", {}).get("date"),
        opponent or raw_json.get("game", {}).get("teams", {}).get("away"),
        raw_json.get("game", {}).get("game_number"),
    )
    if existing_id is not None:
        click.echo(f"Re-importing: deleting existing game id={existing_id}")
        delete_game(conn, existing_id)

    game_id = write_game(
        conn, extraction, str(out_path),
        opponent_override=opponent,
        date_override=date_str,
    )
    conn.close()
    click.echo(f"\nGame id={game_id} written to DB.")

    if do_export:
        click.echo(f"\nExporting season stats to {export_out}...")
        export_season(export_out, db_path=db_path)


if __name__ == "__main__":
    main()
