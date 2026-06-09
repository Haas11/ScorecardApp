from __future__ import annotations

import base64
import json
import os
import re
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from models import GameExtraction

load_dotenv()

_SYSTEM_PROMPT_BASE = """You are a baseball scorecard transcription engine. You will be given an
image of a handwritten Dutch KNBSB baseball scorecard. Extract the data
into structured JSON exactly as specified.

CONFIDENCE RULE: Default to "low" whenever you have any doubt. Only use
"high" when the symbol is completely clear and unambiguous. It is far
better to flag something low-confidence and let a human review it than to
guess wrong silently. A cell with multiple symbols, crossed-out marks, or
unclear handwriting must always be "low".

═══════════════════════════════════════════════════════
TEAM IDENTIFICATION
═══════════════════════════════════════════════════════
The batting team on this scorecard is always "Quick" (the home team).
The opposing (away) team name is written elsewhere on the card — look for
it near the top of the card, in a "Tegenstander" or "Opponent" field, or
next to the game number / date. Do not use "Quick" as the away team.

═══════════════════════════════════════════════════════
CARD LAYOUT — left columns before the scoring grid
═══════════════════════════════════════════════════════
Each row in the lineup has these columns before the scoring boxes:
  1. POS  — fielding position number (1-9; or letters for DH, etc.)
  2. Name — player's last name (and sometimes initial)
  3. Lid.Nr. — jersey / membership number
  4. # (lineup spot) — batting order (1-9)

Read these four columns carefully before reading the scoring cells.
The batting order number is in column 4, not column 1.

═══════════════════════════════════════════════════════
INNING ASSIGNMENT — READ THE HEADER ROW FIRST
═══════════════════════════════════════════════════════
Before reading any PA, look at the header row at the TOP of the scoring
grid. It contains inning numbers (1, 2, 3 … 9 or more). Each column of
scoring cells corresponds to the inning number directly above it in the
header. These are your ground truth.

RULE: The inning number for any PA = the number printed in the header
row above that column. Do NOT guess, increment, or assume sequential
ordering — read the actual printed header value.

SIMPLE CASE (most games): No player's row is completely filled.
The lineup did not cycle fully, so each column maps 1:1 to its header
number. Use the header.

WRAP CASE: A diagonal line through a cell means the lineup wrapped
mid-inning. The cell is split: PAs before the diagonal belong to the
EARLIER inning, PAs after belong to the NEXT inning (header + 1 for
subsequent cells in that row only). Wraps are rare — if you do not
clearly see a diagonal, use the header number as-is.

After assigning innings, count the distinct inning values you used.
Cross-check: if the header shows innings 1–9, you should have at most 9
distinct inning values. If you have invented an inning number not on the
header (e.g. inning 10 when the header only goes to 9), you have made
an error — reassign that PA to the correct header column.

═══════════════════════════════════════════════════════
SUBSTITUTIONS
═══════════════════════════════════════════════════════
When a new player name is written below the original name in the same
batting-order slot, that is a mid-game substitution. Rules:
- Give the original player their own entry in "players[]" with only the
  PAs they recorded before being replaced.
- Give the substitute their own SEPARATE entry in "players[]".
- Record "innings_played" for each: e.g. "1-4" for the starter,
  "5-9" for the sub. The inning number written next to the sub's name
  is the inning they entered.
- The sub's "batting_order" is the same slot number as the starter.
- Do NOT combine both names into a single name string.

═══════════════════════════════════════════════════════
SCORING CELL STRUCTURE — CRITICAL READING ORDER
═══════════════════════════════════════════════════════
Each scoring cell is a small square divided into four triangular corners,
representing the baseball diamond viewed from above:

  TOP-LEFT (3B)  │  TOP-RIGHT (2B)
  ───────────────┼───────────────
  BOT-LEFT (HP)  │  BOT-RIGHT (batter's box)  ◄── ALWAYS START HERE

You MUST read the BOTTOM-RIGHT corner first. That corner and ONLY that
corner contains the primary PA result. The other three corners show
baserunning AFTER the batter reached base and are secondary information.

DO NOT read the top-left, top-right, or bottom-left as the PA result.
If the top-left says "WP" and the bottom-right says "E7", the result is
"E7". The "WP" explains how the runner advanced later — it is not the
PA result.

═══════════════════════════════════════════════════════
STEP 1 — BOTTOM-RIGHT CORNER: THE PA RESULT
═══════════════════════════════════════════════════════

!! ONLY FOUR VALID RESULT VALUES EXIST. NO OTHERS. !!

Every single PA result must be one of these four categories and nothing
else. If you are tempted to write anything outside this list, stop and
re-read the bottom-right corner.

── CATEGORY 1: OUT ───────────────────────────────────
A CIRCLE in the bottom-right corner = OUT. Read what is INSIDE the circle.

Fly outs — letter F followed by a fielder number:
  "F4", "F7", "F8", "F9", etc.
  The F comes BEFORE the number. A circled "4" with no F = groundout,
  not a fly out. Never write "4D", "5D" or any letter after a number.

Groundouts — one or two fielder numbers with a dash:
  "6-3", "4-3", "5-3", "3", "4", "6", etc.

Strikeouts:
  "K"   = swinging strikeout
  "KL"  = called strikeout (backward K)

Other:
  "FC"  = fielder's choice
  "SAC" = sacrifice bunt
  "SF"  = sacrifice fly
  "GDP" = grounded into double play
  "Out" = use ONLY if the symbol inside the circle is truly illegible

── CATEGORY 2: HIT ───────────────────────────────────
The hit symbol is like the letter "I" or "T": a vertical stroke with
horizontal crossbar(s). Count the crossbars:
  1 crossbar → "1B"   (single, including infield hits)
  2 crossbars → "2B"
  3 crossbars → "3B"
  4 crossbars, or ball shown leaving park → "HR"

── CATEGORY 3: WALK ──────────────────────────────────
"BB" written as text in the bottom-right → result: "BB"
Dutch "BB" handwriting can look like "3/3" or a fraction — check carefully.

── CATEGORY 4: HIT BY PITCH ──────────────────────────
"HP" or "HBP" as text in bottom-right → result: "HBP"

E#, WP, PB notations can appear in the three baserunning corners
(top-right, top-left, bottom-left). They then describe how a runner advanced
AFTER reaching base. If you see "E7" in
the top-left corner, that means a runner advanced to 3rd on an error —
it tells you nothing about the current batter's result.
If you find yourself writing "E#" as a result, double check you are reading the right
corner. Look at the bottom-right corner only for PA result. 

═══════════════════════════════════════════════════════
STEP 2 — TOP-RIGHT CORNER: advance to 2nd base
═══════════════════════════════════════════════════════
Only relevant if the batter reached base. Shows HOW they got to 2nd:
SB (stolen base), PB (passed ball), WP (wild pitch), E# (error),
or a hit symbol meaning a teammate's hit advanced them.
Empty = batter did not reach 2nd, or reached 2nd on the original play.

═══════════════════════════════════════════════════════
STEP 3 — TOP-LEFT CORNER: advance to 3rd base
═══════════════════════════════════════════════════════
Same notation as Step 2 but for reaching 3rd base.

═══════════════════════════════════════════════════════
STEP 4 — BOTTOM-LEFT CORNER: run scored
═══════════════════════════════════════════════════════
A FILLED DOT (•) in the center of the 2x2 cell or any mark in the bottom-left corner = run scored.
Set run_scored: true.
A lineup position number may also appear here = that batter gets RBI
credit. Record in notes: e.g. "scored, RBI by batter #3".
If this corner has ANY mark or notation, check carefully whether it
indicates a run scored before concluding run_scored: false.

═══════════════════════════════════════════════════════
BASERUNNING FIELDS
═══════════════════════════════════════════════════════
  sb: count of stolen base events visible in corners 2 and 3 for this PA
  cs: 1 if caught stealing shown, else 0
  notes: describe the baserunning path you see, e.g.:
         "1B, SB to 2B (top-right), WP to 3B (top-left), scored (dot bottom-left)"

═══════════════════════════════════════════════════════
RUN CROSS-CHECK — DO THIS BEFORE OUTPUTTING
═══════════════════════════════════════════════════════
The bottom row of the scorecard shows CUMULATIVE runs per inning
(each number is the running total, not the inning's runs alone).
Convert to per-inning by subtracting: inning_runs[i] = total[i] - total[i-1].

Before you output, count all run_scored=true values across all players.
That count must equal the final cumulative total in the bottom row.
If they disagree, you have missed a run (usually a dot in a bottom-left
corner) or wrongly set run_scored. Correct it before outputting.

═══════════════════════════════════════════════════════
LOGIC CHECKS
═══════════════════════════════════════════════════════
1. OUT + run scored = impossible. A circled result (out) means the player
   never reached base. run_scored must be false. If your reading gives
   both, you have misread the bottom-right corner. Set confidence "low".
2. A batter who made an out has sb=0 and cs=0. Any SB-like marks in the
   cell corners belong to a DIFFERENT baserunner already on base.
3. If run_scored=true, there must be a traceable path home through the
   three base corners. If you cannot trace it, set confidence "low".

═══════════════════════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════════════════════
Return only valid JSON. No text, explanation, or markdown outside the JSON.

{
  "game": {
    "teams": { "away": "<string>", "home": "Quick" },
    "date": "<YYYY-MM-DD or null if illegible>",
    "game_number": "<string or null>"
  },
  "lineup": [
    {
      "batting_order": <int 1-9>,
      "players": [
        {
          "name": "<string>",
          "position": <int or null>,
          "jersey_number": <int or null>,
          "innings_played": "<string or null>",
          "plate_appearances": [
            {
              "inning": <int>,
              "result": "<string>",
              "run_scored": <bool>,
              "rbi": <int>,
              "sb": <int>,
              "cs": <int>,
              "notes": "<string>",
              "confidence": "high" | "low"
            }
          ],
          "summary": {
            "PA":  <int or null>,
            "AB":  <int or null>,
            "R":   <int or null>,
            "H":   <int or null>,
            "2B":  <int or null>,
            "3B":  <int or null>,
            "HR":  <int or null>,
            "BB":  <int or null>,
            "HP":  <int or null>,
            "K":   <int or null>,
            "SB":  <int or null>,
            "CS":  <int or null>,
            "RBI": <int or null>,
            "SAC": <int or null>,
            "SF":  <int or null>
          }
        }
      ]
    }
  ],
  "pitching": [
    {
      "name": "<string>",
      "innings_pitched": <float or null>,
      "runs_allowed": <int or null>,
      "earned_runs": <int or null>,
      "strikeouts": <int or null>,
      "walks": <int or null>,
      "confidence": "high" | "low"
    }
  ],
  "inning_totals": {
    "runs_per_inning": [<int>, ...],
    "errors_total": <int or null>
  },
  "ambiguous_cells": [
    {
      "batter": "<string>",
      "inning": <int>,
      "raw_text": "<string>"
    }
  ]
}"""


def _build_system_prompt(player_names: list[str] | None = None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if player_names:
        names_block = "\n".join(f"  - {n}" for n in player_names)
        prompt += f"""

═══════════════════════════════════════════════════════
KNOWN PLAYER ROSTER
═══════════════════════════════════════════════════════
The following player names are known members of the Quick roster.
When reading a name from the card, match it to the closest name on this
list (accounting for handwriting variation and partial initials).
If a name on the card clearly does not match anyone on the list, still
transcribe it as written and note it in the ambiguous_cells list.

{names_block}
"""
    return prompt


def _archive_path(raw_dir: Path, game_date: str | None, opponent: str | None) -> Path:
    date_str = game_date or date.today().isoformat()
    opp = (opponent or "unknown").lower()
    opp = re.sub(r"[\s]+", "_", opp)
    opp = re.sub(r"[^\w]", "", opp)
    base = raw_dir / f"{date_str}_{opp}.json"
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = raw_dir / f"{date_str}_{opp}_{counter}.json"
        if not candidate.exists():
            return candidate
        counter += 1


def extract_scorecard(
    image_path: str,
    model: str | None = None,
    raw_dir: Path | None = None,
    player_names: list[str] | None = None,
) -> tuple[GameExtraction, Path]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    model = model or os.environ.get("EXTRACTION_MODEL", "claude-sonnet-4-5")
    raw_dir = raw_dir or Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(image_path)
    suffix = img_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image type: {suffix}")

    with open(img_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    system_prompt = _build_system_prompt(player_names)

    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": "Transcribe this scorecard."},
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e

    raw_text = message.content[0].text
    try:
        raw_json = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Try to strip markdown fences if model added them
        stripped = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
        try:
            raw_json = json.loads(stripped)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse VLM response as JSON: {e}") from e

    # Archive raw response
    game_date = raw_json.get("game", {}).get("date")
    teams = raw_json.get("game", {}).get("teams", {})
    opponent = teams.get("away") or teams.get("home")
    archive_path = _archive_path(raw_dir, game_date, opponent)
    archive_path.write_text(json.dumps(raw_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # Remap 2B/3B keys which are not valid Python identifiers in Pydantic
    _remap_summary_keys(raw_json)
    # Drop pitching entries with no name (model placeholder rows)
    if "pitching" in raw_json:
        raw_json["pitching"] = [p for p in raw_json["pitching"] if p.get("name")]

    extraction = GameExtraction.model_validate(raw_json)
    return extraction, archive_path


_STEP1_SYSTEM = """You are a scorecard image analyst. For each inning you OBSERVE the marks and
report them in a FIXED FIELD TEMPLATE. Describe ONLY what you physically see.
Do NOT decide the baseball outcome — a downstream parser does that. Your only
job is to fill the fields accurately.

SCORING GRID STRUCTURE:
Each plate appearance (PA) occupies a 2×2 block divided by a red crosshair:

  [TOP-LEFT]  │ [TOP-RIGHT]
  ────────────┼────────────
  [BOT-LEFT]  │ [BOT-RIGHT]   ← PA result lives in BOT-RIGHT; run indicator in BOT-LEFT

The red vertical line divides LEFT from RIGHT.
The red horizontal line divides TOP from BOTTOM.
A PRINTED inning-number digit sits in the header row directly above each column.

════════════════════════════════════════
STEP A — ANCHOR THE INNING NUMBERS (do this FIRST)
════════════════════════════════════════
Read the PRINTED header digits left to right and list them:
  "Header digits: 1 2 3 4 5 6 7 8 9"
Every inning block below is tagged with the printed digit DIRECTLY ABOVE that
column. NEVER number blocks by counting position — always read the printed
digit overhead. This prevents off-by-one shifts.
Skip any column whose header reads 10, 11, 12, PA, AB, H, R, AVG, etc.

════════════════════════════════════════
KEY FACTS ABOUT THE SYMBOLS (read before filling fields)
════════════════════════════════════════
1. CIRCLE LOCATION DECIDES MEANING. A circle is the PA RESULT (an OUT) ONLY when
   it FILLS THE BOT-RIGHT quadrant. The character(s) inside identify the out type.
   BUT a circle drawn ON A BASE — i.e. up in a TOP quadrant, or centered on the
   diamond/crosshair, or anywhere that is NOT the BOT-RIGHT batter's box — is a
   BASERUNNING OUT (a runner was tagged/forced out later). That circle is NOT the
   PA result. When you see a circle on a base AND a separate mark down in
   BOT-RIGHT, the BOT-RIGHT mark is the PA result; the on-base circle is secondary.
   (A walk or a hit can never be inside the BOT-RIGHT out-circle — you cannot have
   a walk AND an out in the same PA.)

2. "BB" (walk) is TWO adjacent rounded humps written as TEXT in BOT-RIGHT, with
   NO large circle around them. Handwritten BB is EASILY misread as "13", "16",
   "B6", "8", "13-3", or "B3". Whenever you see two rounded pen humps in
   BOT-RIGHT that are NOT enclosed by a circle, seriously consider that it is
   "BB" and report it as such. It will NEVER be "3-3" (not a real result).
   ("PB" is a passed ball — that is a baserunning note, not a PA result.)

3. "FC" (fielder's choice) is ALWAYS a PA result and ALWAYS lives in BOT-RIGHT.
   It is the outcome of a batted ball. Never treat FC as a baserunning note.

4. A "K" (strikeout) is USUALLY an out. RARE EDGE CASE: on a dropped third
   strike / passed ball the batter strikes out BUT still reaches base. In that
   case you will see a "K" AND a run/base-reached indicator together — report
   both fields exactly as you see them; do not suppress one.

5. RUN-SCORED HAS TWO POSSIBLE INDICATORS, either of which counts:
   (a) any mark in the BOT-LEFT sub-cell, and/or
   (b) a black dot at the CENTER crosshair point where the red lines meet.
   Report each in its own field below.

════════════════════════════════════════
STEP B — DESCRIBE EACH INNING IN THIS EXACT TEMPLATE
════════════════════════════════════════
Output one block per inning, ALL 9 innings, in this format. No prose, no
interpretation — just fill each field:

INNING <header digit>:
  CIRCLE: <none | large circle filling BOT-RIGHT (PA out) | circle on a base/top/center (baserunning out, NOT the result)>
  CIRCLE_CONTENTS: <exact characters inside the circle, or n/a>
  HIT_STROKE: <none | vertical stroke with N crossbar(s) in BOT-RIGHT>
  BB_BOT_RIGHT: <none | present — two rounded humps as the main BOT-RIGHT mark, no circle>
  BOT_RIGHT_TEXT: <none | exact characters NOT in a circle, e.g. "E7", "FC", "K">
  TOP-LEFT: <empty | X present | other: "...">
  TOP-RIGHT: <empty | X present | other: "...">
  X_COUNT: <total number of X marks in the top quadrants: 0, 1, or 2>
  BOT-LEFT: <BLANK | MARK PRESENT: describe (dot / diagonal tick / number / stroke)>
  CENTER_DOT: <none | black dot at the center crosshair>
  TEMPLATE_DIAGONAL: <yes | no>

FIELD RULES:

• CIRCLE / CIRCLE_CONTENTS — judge WHERE the circle sits (see KEY FACT 1).
  If it fills the BOT-RIGHT batter's box → "large circle filling BOT-RIGHT
  (PA out)" and transcribe its contents (K, F7, 6-3, 54, etc.) — the result is
  an OUT regardless of contents. If the circle sits on a base (a top quadrant)
  or is centered on the diamond → "circle on a base/top/center (baserunning out,
  NOT the result)" and still transcribe its contents, but do NOT treat it as the
  PA result — keep looking in BOT-RIGHT for the real result mark.

• HIT_STROKE — a vertical stroke "|" crossed by horizontal bars. Count the bars.
  Only fill this when there is NO large circle.

• BB_BOT_RIGHT — see KEY FACT 2. Fill "present" only when the two humps are the
  main BOT-RIGHT mark and there is NO large circle around them.

• BOT_RIGHT_TEXT — letters/numbers in BOT-RIGHT that are NOT inside a circle,
  NOT a hit stroke, and NOT BB (e.g. "E7", "FC", "HBP"). Quote exactly.

• TOP-LEFT / TOP-RIGHT / X_COUNT — base-reached markers for hits. Report each X,
  then give the TOTAL count in X_COUNT. The count maps to the base reached:
  0 X = 1st base (1B); 1 X = 2nd base (2B); 2 X = 3rd base (3B).
  Look CAREFULLY for a second X — it is easy to miss the triple.

• BOT-LEFT — RUN-SCORED INDICATOR (a), OFTEN FAINT. Look specifically at the
  lower-left sub-cell. It is commonly a short diagonal pen tick or a small filled
  dot, lighter than the result mark. Do NOT default to BLANK — only write BLANK
  if the lower-left sub-cell is genuinely clean. If you see ANY pen mark there,
  even faint, write "MARK PRESENT" and describe it.

• CENTER_DOT — RUN-SCORED INDICATOR (b). Look at the exact CENTER point where the
  red vertical and horizontal lines cross. A filled black dot there also means a
  run scored. Report "black dot at the center crosshair" or "none".

• TEMPLATE_DIAGONAL — "yes" ONLY when a single SW→NE line crosses an otherwise
  empty block (CIRCLE none, HIT_STROKE none, BB none, BOT_RIGHT_TEXT none). That
  inning is an unused/empty inning, no PA.

════════════════════════════════════════
STEP C — SUMMARY LINES (write after all 9 blocks)
════════════════════════════════════════
  "PA innings: [header digits where CIRCLE / HIT_STROKE / BB_BOT_RIGHT / BOT_RIGHT_TEXT is non-empty]"
  "Run innings: [header digits where BOT-LEFT = MARK PRESENT or CENTER_DOT present]"

PER PLAYER: state name and batting order (POS | Name | Lid.Nr. | order#).
Also describe the game header, any substitutions, and the bottom score row.
Do not produce JSON."""

_STEP2_SYSTEM = """You are a baseball scorecard parser. Each inning arrives as a block of LABELED
FIELDS produced by an image analyst. Map the fields to JSON MECHANICALLY. Do NOT
re-interpret the image — trust the fields exactly as written.

TEAM: home team is always "Quick". Away team is stated in the game header.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INNING NUMBERS — ABSOLUTE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each block is headed "INNING <n>:". Use exactly that digit. NEVER renumber,
compress, reorder, or shift innings.

Create a PA entry ONLY when the block has PA-content (see PA RESULT below) OR
BOT-LEFT = MARK PRESENT. If the block is a TEMPLATE_DIAGONAL: yes, or all result
fields are none/empty and BOT-LEFT is BLANK → omit that inning entirely.
Discard any PA with inning > 9.

VALIDATION: cross-check every PA against the analyst's "PA innings: [...]" line.
If you would add a PA for an inning NOT in that list, omit it.

SUBSTITUTIONS: second name in the same batting slot = two separate entries in
"players[]" with the same batting_order.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PA RESULT — decide in THIS ORDER, stop at the first match
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BB OVERRIDE — if the letters "BB" appear ANYWHERE in the block (BB_BOT_RIGHT
   = present, or "BB" in BOT_RIGHT_TEXT, or "BB" noted near/below a circle or in
   BOT-LEFT) AND "BB" is NOT the CIRCLE_CONTENTS itself → result = "BB" (walk).
   A walk cannot coexist with an out, so any circle in the block is then a
   BASERUNNING out, not the PA result. (Exclusion: if the only BB-like reading is
   INSIDE the circle as CIRCLE_CONTENTS, that is a misread of an out — do NOT
   call it a walk; treat it as the out in step 2.)

2. CIRCLE = "large circle filling BOT-RIGHT (PA out)"  → the batter made an OUT.
   Read CIRCLE_CONTENTS for the out TYPE (the result is OUT regardless of
   contents):
     "K" / swinging                  → "K"
     backward K / "KL" / called      → "KL"
     "F" + number                    → e.g. "F7"   (fly out)
     number-dash-number              → e.g. "6-3"  (groundout)
     two digits no dash (e.g. 54)    → "5-4"       (groundout)
     single digit                    → e.g. "3"
     "FC" / "SAC" / "GDP"            → as written
     illegible / n/a                 → "Out"
   DO NOT treat a CIRCLE of "circle on a base/top/center (baserunning out)" as an
   out here — it is NOT the PA result. Skip it and continue to step 3 to find the
   real result down in BOT-RIGHT.

3. HIT — fire this if EITHER a HIT_STROKE with AT LEAST ONE crossbar is present,
   OR X_COUNT >= 1 (X marks mean the batter reached 2nd/3rd, which only happens
   on a hit/reach). A "vertical stroke with 0 crossbar(s)" alone is NOT a hit.
   Base reached comes from X_COUNT:
     X_COUNT = 0  → "1B"
     X_COUNT = 1  → "2B"
     X_COUNT = 2  → "3B"
     HR notation / ball left park → "HR"

4. BOT_RIGHT_TEXT non-empty  → map the text:
     "K"                 → "K"    (strikeout — an OUT, even with no circle)
     "KL" / backward K   → "KL"   (called strikeout — an OUT)
     "HP" / "HBP"        → "HBP"
     "E" + digit         → e.g. "E7"  (reached on error)
     "FC"                → "FC"       (fielder's choice — always a result)
     "SAC"               → "SAC"
     anything unclear    → best guess, confidence "low"
   BASERUNNING-ONLY NOTATIONS ARE NOT RESULTS: "PB" (passed ball), "SB" (stolen
   base), "WP" (wild pitch), "CS" (caught stealing) describe how a runner moved,
   never how the batter's PA ended. If BOT_RIGHT_TEXT is ONLY one of these (and
   there is no circle, hit, BB, error, FC, or HBP) → this is NOT a PA result.
   Go to step 5 and omit the PA.

5. None of the above  → no PA this inning (omit it).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN SCORED — read ONLY the BOT-LEFT field
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A run scored if EITHER indicator is present: BOT-LEFT = "MARK PRESENT", and/or
CENTER_DOT = present (black dot at the center crosshair). Either one alone is
enough.

RULE A — if the RESULT is an OUT → run_scored = false, ALWAYS. Ignore the run
  indicators. An OUT is: anything from a large BOT-RIGHT circle (step 2), OR a
  strikeout "K"/"KL" from text (step 4). A batter who is out cannot score.

RULE B — otherwise (BB, hit, E#, HBP, FC):
  BOT-LEFT = "MARK PRESENT" OR CENTER_DOT present  → run_scored = true
  both BLANK / none                                → run_scored = false
  Never infer a run from BOT_RIGHT_TEXT or the top quadrants.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PA COUNT ORDERING — MANDATORY FINAL ENFORCEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseball rule: the lead-off batter (order 1) always gets the MOST plate
appearances. Each successive batter gets the same count OR FEWER — never more.

  PA_count(1) >= PA_count(2) >= PA_count(3) >= ... >= PA_count(9)

This is a HARD CONSTRAINT. You MUST enforce it before outputting JSON.

ENFORCEMENT PROCEDURE — execute this for every batting position N from 2 to 9:
  1. Count PAs for position N and position N-1.
  2. If count(N) > count(N-1):
       a. Calculate excess = count(N) - count(N-1).
       b. From position N's PA list, find the 'excess' entries with the
          LOWEST confidence (prefer confidence="low"; if tied, prefer the
          highest inning number as it is most likely a phantom from a bleed
          or inning-shift error).
       c. DELETE those entries from the JSON.
       d. Recount and repeat until count(N) <= count(N-1).
  3. Do NOT output the JSON until all positions satisfy the rule.

Exception: if a batting slot contains TWO players (substitution), sum their
PA counts together for the ordering check — do not check them individually.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OTHER FIELDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set sb=0, cs=0, rbi=0 for every PA (not tracked).
confidence = "high" when the analyst described a clear, unambiguous symbol.
confidence = "low" when the analyst used words like "unclear", "possibly",
  "looks like", "hard to read", or described conflicting symbols, or when
  BOT-RIGHT contained multiple items that needed a priority rule to resolve.

OUTPUT: return only valid JSON — no text, no markdown outside the JSON.

{
  "game": {
    "teams": { "away": "<string>", "home": "Quick" },
    "date": "<YYYY-MM-DD or null>",
    "game_number": "<string or null>"
  },
  "lineup": [
    {
      "batting_order": <int 1-9>,
      "players": [
        {
          "name": "<string>",
          "position": <int or null>,
          "jersey_number": <int or null>,
          "innings_played": "<string or null>",
          "plate_appearances": [
            {
              "inning": <int>,
              "result": "<string>",
              "run_scored": <bool>,
              "rbi": <int>,
              "sb": <int>,
              "cs": <int>,
              "notes": "<string>",
              "confidence": "high" | "low"
            }
          ],
          "summary": {
            "PA": <int or null>, "AB": <int or null>,
            "R": <int or null>, "H": <int or null>,
            "2B": <int or null>, "3B": <int or null>,
            "HR": <int or null>, "BB": <int or null>,
            "HP": <int or null>, "K": <int or null>,
            "SB": <int or null>, "CS": <int or null>,
            "RBI": <int or null>, "SAC": <int or null>,
            "SF": <int or null>
          }
        }
      ]
    }
  ],
  "pitching": [],
  "inning_totals": {
    "runs_per_inning": [<int>, ...],
    "errors_total": <int or null>
  },
  "ambiguous_cells": [
    { "batter": "<string>", "inning": <int>, "raw_text": "<string>" }
  ]
}"""


def _make_image_content(image_data: str, media_type: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": image_data},
    }


def extract_scorecard_two_step(
    image_path: str,
    model: str | None = None,
    raw_dir: Path | None = None,
    player_names: list[str] | None = None,
) -> tuple[GameExtraction, Path]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    model = model or os.environ.get("EXTRACTION_MODEL", "claude-sonnet-4-5")
    raw_dir = raw_dir or Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(image_path)
    suffix = img_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image type: {suffix}")

    with open(img_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)

    # ── Step 1: visual description ────────────────────────────────────
    print("  Step 1/2: visual grid scan...")
    roster_hint = ""
    if player_names:
        roster_hint = (
            "\n\nKNOWN ROSTER (Quick): "
            + ", ".join(player_names)
            + "\nMatch player names from the card to this list where possible."
        )
    try:
        step1_msg = client.messages.create(
            model=model,
            max_tokens=8192,
            system=_STEP1_SYSTEM + roster_hint,
            messages=[{
                "role": "user",
                "content": [
                    _make_image_content(image_data, media_type),
                    {"type": "text", "text": (
                        "Describe every cell on this scorecard as instructed. "
                        "Red lines are drawn inside each PA block — they are your "
                        "exact left/right and top/bottom dividers. Check which side "
                        "of BOTH red lines each marking falls on before labeling it "
                        "BOT-RIGHT, BOT-LEFT, TOP-RIGHT, or TOP-LEFT."
                    )},
                ],
            }],
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error (step 1): {e}") from e

    description = step1_msg.content[0].text
    print(f"  Step 1 complete ({len(description)} chars). Running step 2...")

    # ── Step 2: parse description into JSON ──────────────────────────
    try:
        step2_msg = client.messages.create(
            model=model,
            max_tokens=8192,
            system=_STEP2_SYSTEM,
            messages=[{
                "role": "user",
                "content": description,
            }],
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error (step 2): {e}") from e

    raw_text = step2_msg.content[0].text
    try:
        raw_json = json.loads(raw_text)
    except json.JSONDecodeError:
        stripped = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
        try:
            raw_json = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse step-2 response as JSON: {e}") from e

    # Archive both the description and the final JSON
    game_date = raw_json.get("game", {}).get("date")
    teams = raw_json.get("game", {}).get("teams", {})
    opponent = teams.get("away") or teams.get("home")
    archive_path = _archive_path(raw_dir, game_date, opponent)
    archive_path.write_text(
        json.dumps({"_step1_description": description, **raw_json}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _remap_summary_keys(raw_json)
    if "pitching" in raw_json:
        raw_json["pitching"] = [p for p in raw_json["pitching"] if p.get("name")]

    extraction = GameExtraction.model_validate(raw_json)
    return extraction, archive_path


def _remap_summary_keys(raw: dict) -> None:
    """Rename 2B/3B summary keys to two_B/three_B for Pydantic."""
    for slot in raw.get("lineup", []):
        for player in slot.get("players", []):
            summary = player.get("summary")
            if summary:
                if "2B" in summary:
                    summary["two_B"] = summary.pop("2B")
                if "3B" in summary:
                    summary["three_B"] = summary.pop("3B")
