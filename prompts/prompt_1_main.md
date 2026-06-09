# Prompt 1 — Main build prompt
# Paste this in full into a new Claude Code session

---

Build a baseball scorecard digitization and season stats tracking system in Python.

## Overview

A CLI pipeline that:
1. Takes a photo of a handwritten Dutch KNBSB baseball scorecard
2. Calls the Anthropic Claude API via vision to extract play-by-play data
3. Validates and stores results in a local SQLite database
4. Computes and exports season-level player statistics to Excel

## Project structure

```
scorecard/
  process_game.py       # CLI entrypoint
  extract.py            # VLM extraction via Anthropic API
  models.py             # Pydantic schemas
  db.py                 # SQLite setup and write layer
  stats.py              # Stat computation
  export_season.py      # Excel export
  review.py             # Interactive low-confidence PA correction CLI
  manage_players.py     # Player alias management CLI
  config.yml            # League config (avg stats, thresholds)
  .env                  # API key and model config (never commit)
  .env.example          # Template with all required keys
  data/
    season.db           # SQLite database (created on first run)
    raw/                # Archived raw VLM JSON responses
  README.md
```

## Dependencies

Use uv. Required packages:
  anthropic, pydantic, openpyxl, pillow, click, rapidfuzz,
  python-dotenv, pyyaml

## Environment variables (.env)

```
ANTHROPIC_API_KEY=your_key_here
EXTRACTION_MODEL=claude-sonnet-4-20250514
# Switch to claude-opus-4-20250514 for difficult/dense handwriting
```

## Configuration file (config.yml)

```yaml
league:
  avg_ops: 0.720          # Update after first full season of data
  avg_obp: 0.320
  avg_woba: 0.320
  woba_weights:
    bb: 0.69
    hp: 0.72
    single: 0.89
    double: 1.27
    triple: 1.62
    hr: 2.10
stats:
  small_sample_threshold: 10   # PA below this: flag rate stats
fuzzy:
  auto_match_threshold: 90     # score >= this: silent auto-match
  warn_threshold: 70           # score >= this: match with warning
```

## VLM extraction (extract.py)

Load EXTRACTION_MODEL from .env. Call the Anthropic API with the
scorecard image base64-encoded. Detect media type from file extension
(jpeg/png). Handle API errors with informative messages.

Before parsing, archive the raw JSON response to:
  data/raw/<YYYY-MM-DD>_<opponent_normalized>.json
where opponent_normalized is the opponent name lowercased with spaces
replaced by underscores. If the file already exists, append a counter
suffix (_2, _3 etc.) to avoid overwriting.

Use this system prompt verbatim:

---
You are a baseball scorecard transcription engine. You will be given an
image of a handwritten Dutch KNBSB baseball scorecard. Extract the data
into structured JSON exactly as specified. Do not infer or guess plays —
if a cell is illegible or ambiguous, set "confidence": "low" and
transcribe what you can see.

NOTATION REFERENCE (Dutch KNBSB):
- K / backward-K = strikeout (swinging / looking)
- BB = walk
- HP = hit by pitch
- 1B / 2B / 3B / HR = hit types
- E# = error by fielder # (e.g. E7 = error by left fielder)
- FC = fielder's choice
- SB = stolen base
- CS = caught stealing
- PB = passed ball
- WP = wild pitch
- F# = fly out to fielder #
- #-# = groundout or forceout (e.g. 6-3 = shortstop to first base)
- SAC / SH = sacrifice bunt
- SF = sacrifice fly
- GDP = grounded into double play
- IO = infield out
- Li = line drive out (sometimes written L or Li)
- A circle drawn around a play symbol = run scored on that plate appearance
- Numbers in cell corners = baserunner tracking (which base reached)
- Diagonal lines in a cell = inning boundary marker
- A new player name written directly below the original player in the
  same batting order slot = mid-game substitution. Note the inning number
  written beside the substitute's name.

INNING ASSIGNMENT:
When a diagonal line appears in a cell, the plate appearance belongs to
the inning in which it was completed (the inning after the diagonal).
Use diagonal lines to correctly assign inning numbers to all PAs.

OUTPUT SCHEMA:
Return only valid JSON. No text, explanation, or markdown outside the JSON.

{
  "game": {
    "teams": { "away": "<string>", "home": "<string>" },
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
}
---

## Pydantic models (models.py)

Mirror the JSON schema above exactly. Add a model-level validator on
GameExtraction that checks: if summary.PA is not null, verify that
len(plate_appearances) <= summary.PA (not strictly equal, since PAs
can span substitutions). Flag mismatches as a warning, not an error.
Do not reject the extraction over a summary mismatch.

## Database schema (db.py)

Tables:

```sql
players(
  player_id   INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  jersey_number   INTEGER
)

player_aliases(
  alias           TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  player_id       INTEGER REFERENCES players(player_id)
)

games(
  game_id     INTEGER PRIMARY KEY,
  date        TEXT,
  opponent    TEXT,
  game_number TEXT,
  raw_json_path TEXT,
  imported_at TEXT,
  UNIQUE(date, opponent, game_number)
)

plate_appearances(
  pa_id         INTEGER PRIMARY KEY,
  player_id     INTEGER REFERENCES players(player_id),
  game_id       INTEGER REFERENCES games(game_id),
  inning        INTEGER,
  batting_order INTEGER,
  result        TEXT,
  run_scored    INTEGER,
  rbi           INTEGER,
  sb            INTEGER,
  cs            INTEGER,
  bb            INTEGER,
  hp            INTEGER,
  sac           INTEGER,
  sf            INTEGER,
  raw_notes     TEXT,
  confidence    TEXT,
  needs_review  INTEGER DEFAULT 0,
  reviewed      INTEGER DEFAULT 0
)

pitching_appearances(
  pa_id           INTEGER PRIMARY KEY,
  player_id       INTEGER REFERENCES players(player_id),
  game_id         INTEGER REFERENCES games(game_id),
  innings_pitched REAL,
  runs_allowed    INTEGER,
  earned_runs     INTEGER,
  strikeouts      INTEGER,
  walks           INTEGER,
  confidence      TEXT,
  needs_review    INTEGER DEFAULT 0,
  reviewed        INTEGER DEFAULT 0
)
```

Set needs_review=1 for any PA or pitching appearance where confidence="low".

## Fuzzy name matching (db.py)

Use rapidfuzz.fuzz.token_sort_ratio for all player lookups.

Resolution order when a name arrives from extraction:
1. Exact match on normalized_name
2. Check player_aliases.normalized_alias for exact match
3. Fuzzy match on normalized_name using token_sort_ratio:
   - Score >= auto_match_threshold (from config.yml): silent auto-match,
     log to console: "Auto-matched '{incoming}' → '{stored}' ({score})"
   - Score >= warn_threshold and < auto_match_threshold: match with warning:
     "Fuzzy match: '{incoming}' matched to '{stored}' (score: {score})
      — run 'python manage_players.py --aliases' to review"
   - Score < warn_threshold: insert as new player

Normalize names by: lowercase, strip leading/trailing whitespace,
collapse multiple spaces, remove dots and hyphens.
Store normalized_name on insert.

## CLI — process_game.py

```
Usage: python process_game.py <image_path>
         [--opponent TEXT]     # override extracted opponent name
         [--date TEXT]         # override extracted date (YYYY-MM-DD)
         [--dry-run]           # extract and validate but do not write to DB
         [--review]            # show PA table and prompt confirmation before writing
         [--model TEXT]        # override EXTRACTION_MODEL from .env
```

Execution flow:
1. Load image, base64-encode
2. Call extract.py → raw JSON
3. Archive raw JSON to data/raw/
4. Validate with Pydantic models
5. Apply --opponent and --date overrides if provided
6. Check for duplicate game in DB (same date + opponent + game_number)
   - If duplicate found: print warning and prompt "Re-import? This will
     delete existing PAs for this game. [y/n]"
   - On y: delete existing game record and PAs, then proceed
   - On n: exit cleanly
7. Print summary of extraction:
   - List of players found with PA counts
   - List of any low-confidence cells grouped by player
   - List of ambiguous_cells
   - Fuzzy match warnings if any
8. If --review: print full PA table per player, prompt "Confirm import? [y/n]"
9. Write to DB via db.py
10. Print one-line stat summary per player (AB-H AVG)

## CLI — review.py

```
Usage: python review.py [--game DATE] [--player NAME]
```

- Query all plate_appearances where needs_review=1 and reviewed=0
- Display each one with context: player name, game date, opponent, inning,
  result, raw_notes
- Prompt: "Correct result (or Enter to keep '<current>'):"
- On correction: update result in DB
- Mark reviewed=1 regardless of whether a correction was made
- At end: print count of reviewed PAs

## CLI — manage_players.py

```
Usage: python manage_players.py --aliases   # review pending fuzzy matches
       python manage_players.py --list      # list all players and aliases
       python manage_players.py --merge NAME1 NAME2  # manually merge two players
```

--aliases:
  Show all auto-matches and warn-matches from last import (store these
  in a pending_aliases temp table cleared after review).
  For each: "Are '{incoming}' and '{stored}' the same player? [y/n]"
  On y: add to player_aliases
  On n: create new player record, reassign PAs

--merge NAME1 NAME2:
  Merge all PAs from NAME2 into NAME1, delete NAME2 record,
  add NAME2 as alias of NAME1.

## Stat computation (stats.py)

Load woba_weights and league averages from config.yml.

For each player compute over all plate_appearances:

Standard:
  G    = count of distinct game_ids
  PA   = count of all PAs
  AB   = PA - BB - HP - SAC - SF
  H    = count where result IN ('1B','2B','3B','HR')
  2B   = count where result = '2B'
  3B   = count where result = '3B'
  HR   = count where result = 'HR'
  R    = sum of run_scored
  RBI  = sum of rbi
  BB   = sum of bb
  K    = count where result = 'K'
  SB   = sum of sb
  CS   = sum of cs
  AVG  = H / AB  (0 if AB=0)
  OBP  = (H + BB + HP) / (AB + BB + HP + SF)  (0 if denominator=0)
  SLG  = (1B + 2*2B + 3*3B + 4*HR) / AB  (0 if AB=0)
  OPS  = OBP + SLG

Advanced:
  BABIP    = (H - HR) / (AB - K - HR + SF)  (null if denominator=0)
  ISO      = SLG - AVG
  BB_pct   = BB / PA
  K_pct    = K / PA
  wOBA     = (w_bb*BB + w_hp*HP + w_1b*1B + w_2b*2B + w_3b*3B + w_hr*HR)
             / (AB + BB + SF + HP)  (null if denominator=0)
  RC       = (H + BB) * (TB) / (AB + BB)  where TB = 1B + 2*2B + 3*3B + 4*HR
  OPS_plus = 100 * (OBP/lg_obp + SLG/lg_slg - 1)  (100 = league average)
  AB_per_HR = AB / HR  (null if HR=0)
  BB_K     = BB / K  (null if K=0)

Flag small_sample=True if PA < small_sample_threshold from config.yml.

Return a dataclass or TypedDict per player. Never store derived stats in DB.

## Export (export_season.py)

```
Usage: python export_season.py [--output stats.xlsx] [--min-pa INT]
```

- Query all players, compute stats via stats.py
- Filter to players with PA >= --min-pa (default 0, so all players included)
- Sort by OPS descending
- Write to Excel with openpyxl:

Sheet 1 "Season Stats":
  - One row per player
  - Columns: Name, G, PA, AB, H, 2B, 3B, HR, R, RBI, BB, K, SB, CS,
             AVG, OBP, SLG, OPS, BABIP, ISO, BB%, K%, wOBA, RC, OPS+,
             AB/HR, BB/K
  - Header row: bold, frozen, background fill #1F4E79, white font
  - AVG/OBP/OPS/wOBA columns: green gradient conditional formatting
    white at league average, green (#C6EFCE) at +0.050 above average
  - Rate stats (AVG, OBP, SLG, OPS, BABIP, ISO, wOBA): 3 decimal places
  - Counting stats: integers
  - small_sample players: italicize entire row
  - Auto-fit column widths

Sheet 2 "Game Log":
  - One row per player per game
  - Columns: Name, Date, Opponent, PA, AB, H, 2B, 3B, HR, R, RBI, BB, K,
             SB, CS, AVG (game), OBP (game)
  - Sorted by Date ascending, then batting order

Sheet 3 "Low Confidence":
  - All PAs where needs_review=1
  - Columns: Player, Date, Opponent, Inning, Result, Notes, Reviewed
  - So the coaching staff can see what needs attention

## README.md

Include:
- Prerequisites (Python 3.11+, uv, Node for Claude Code)
- Setup instructions (uv sync, .env setup, first run)
- Usage examples for all CLI commands
- Explanation of fuzzy matching thresholds and how to tune them
- Note on model selection: sonnet for clean cards, opus for dense/messy ones
