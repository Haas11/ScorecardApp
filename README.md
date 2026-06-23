# KNBSB Scorecard Pipeline

Digitizes Dutch KNBSB baseball scorecards into structured JSON, a SQLite database, and an Excel workbook.

## Setup

```powershell
cd scorecard
uv sync
```

Create `scorecard/.env` with your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...          # for Gemini (recommended)
EXTRACTION_MODEL=gemini-2.5-flash
```

---

## Image naming

Name scan files as `YYYY-MM-DD_<opponent>.jpg` (or `.png`) and place them in `images/scans/`:

```
images/scans/
  2026-06-07_almere.jpg
  2026-06-14_quick.jpg
```

Date and opponent are parsed automatically from the filename.

---

## Processing a game — step by step

All commands run from the `scorecard/` directory.

### Step 1 — Run the extraction

```powershell
uv run python extract_cells.py "../images/scans/YYYY-MM-DD_<opponent>.jpg" `
  --players ../players.txt `
  --innings 9
```

The pipeline:
1. Detects the grid (row/column boundaries)
2. Classifies each PA cell via VLM (Gemini by default)
3. Applies structural rules (isolation, 3-out, K-PB)
4. Cross-checks against ground-truth totals if present

Output goes to `images/data/raw/{stem}_cells.json`.

### Step 2 — Check the output

Read the terminal output. Look for:

- `MISMATCH` lines in the per-player or per-inning checks
- `[isolated]` or `[3-outs]` removals — confirm they look right
- `Outs:WARNING` in per-inning check is **expected** (runner outs from FC/SB aren't in PA results)

### Step 3 — Edit ground-truth files if needed

Two files are auto-generated in `images/ground_truth/`. Open and correct any misread totals:

**`{stem}_totals.txt`** — per-inning team totals from the card's bottom strip:
```
# inning  runs  hits  errors  lob
1  3  1  2  0
2  0  2  0  2
```

**`{stem}_stats.txt`** — per-player H and AB from the stats column:
```
# slot  H  AB
1  3  4
5  1  4
5  1  1     ← second row for same slot = substitution
```

### Step 4 — Re-run with cache

```powershell
uv run python extract_cells.py "../images/scans/YYYY-MM-DD_<opponent>.jpg" `
  --players ../players.txt --innings 9 --reuse-cache
```

`--reuse-cache` skips API calls for cells already classified. Structural rules always re-run from scratch (removed cells are restored and re-evaluated each run).

Repeat steps 2–4 until all cross-checks show OK.

### Step 5 — Inspect individual cells (optional)

```powershell
uv run python _dump_cells.py > cells.csv
```

Writes a CSV with one row per (player, inning): result, run, confidence, notes.

---

## Roster file

`players.txt` lists one player per batting slot (name, jersey number):

```
Max. Gelaudi, 98
Joey Bradwell, 71
Victor van Spaandonk, 24
```

Substitutions share the same batting slot and are NOT added as separate rows.

---

## Players and aliases

After import, confirm fuzzy-matched names:

```powershell
uv run python manage_players.py aliases
uv run python manage_players.py list
```

---

## Export to Excel

```powershell
uv run python export_season.py --output ../stats.xlsx
```

| Sheet | Contents |
|---|---|
| Season Stats | Cumulative stats for all players, sorted by OPS |
| Game Log | Per-player per-game line |
| {Date} {Opponent} | Box score for each game |
| Low Confidence | PAs flagged for review |

---

## Review low-confidence PAs

```powershell
uv run python review.py
uv run python review.py --game 2026-06-07
```

After reviewing:

```powershell
uv run python mark_reviewed.py --date 2026-06-07 --opponent almere
```

---

## File layout

```
images/
  scans/              ← drop scan images here
  ground_truth/       ← hand-correct after each run
    {stem}_totals.txt
    {stem}_stats.txt
  _cache/             ← generated; never edit manually; gitignored
    cells/{stem}/     ← per-cell VLM results (JSON, 1 per cell)
  data/
    raw/              ← per-game GameExtraction JSON

scorecard/
  extract_cells.py    ← main pipeline (new, per-cell)
  process_game.py     ← old pipeline (full-image VLM, still works)
  export_season.py    ← Excel export (reads DB)
  review.py           ← review low-confidence PAs
  mark_reviewed.py    ← mark PAs as reviewed
  manage_players.py   ← player alias management
  data/
    season.db         ← SQLite database
    raw/              ← per-game JSON archives

players.txt           ← roster (one player per batting slot)
stats.xlsx            ← exported workbook
PIPELINE.md           ← backend architecture reference
```

---

## Notes

- **API key safety**: `.env` is gitignored. Never commit API keys.
- **DB integration**: `extract_cells.py` outputs JSON but does not yet write to `season.db`. Use `process_game.py` for DB/export until that bridge is built. See [PIPELINE.md](PIPELINE.md) for details.
- **Re-running a game**: run the same command again. The pipeline detects the existing game, replaces it, and re-exports. Season totals are never double-counted.
