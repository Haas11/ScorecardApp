# KNBSB Scorecard Pipeline

Digitizes Dutch KNBSB baseball scorecards into a SQLite database and exports a shared Excel workbook.

## Setup

```powershell
cd scorecard
uv sync
```

Requires an Anthropic API key in `scorecard/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Analyzing a New Game

All commands run from the `scorecard/` directory.

You need **one image file** per game: a scan with hand-drawn red divider lines and the per-player stats column included on the right.

Add `--no-stats` to skip per-player H-AB extraction for games that don't have a stats column.

---

### Step 1 — Extract and review

```powershell
uv run python extract_rows.py "../images/score cards with grid/<GAME>.png" `
  --enhance esrgan --innings 9 --players 9 `
  --players-file ../players.txt `
  --date YYYY-MM-DD --opponent <NAME> `
  --realign --dry-run
```

Read the terminal output. Check:

- **`[LOW]` plays** — cells the model was uncertain about; compare against the physical card
- **Reconciliation mismatches** — `MISSED run(s)`, `EXTRA E#`, etc. at the bottom of the output

If a mismatch is caused by the totals strip being misread (wrong run/hit/error/LOB count), correct it in the auto-generated cache file and re-run (step 2). If individual play results are wrong, note them — the constraints will fix most automatically on re-run.

---

### Step 2 — Correct misread totals (only if needed)

Two text files are auto-generated during step 1. Open them in any text editor and fix misread numbers:

**`../images/totals_cache/<GAME>_totals.txt`** — per-inning team totals from the card's bottom strip:

```
# inning  runs  hits  errors  lob
1    3    1    2    0
2    0    2    0    2
```

**`../images/player_stats_cache/<GAME>.txt`** — per-player H and AB from the stats column:

```
# slot  H  AB
1    3    4
2    0    5
```

You only need to edit these if the numbers don't match the physical card.

---

### Step 3 — Re-run until clean

```powershell
uv run python extract_rows.py "../images/score cards with grid/<GAME>.png" `
  --enhance esrgan --reuse-step1 --innings 9 --players 9 `
  --players-file ../players.txt `
  --date YYYY-MM-DD --opponent <NAME> `
  --stats-image "../images/score cards/<GAME clean>.jpg" `
  --realign --dry-run
```

`--reuse-step1` replays the cached vision descriptions — no API calls, instant re-run. Repeat steps 2–3 until the output shows:

```
Run reconciliation OK
Hit reconciliation OK
PA count OK
Lineup continuity OK
```

---

### Step 4 — Import to database and export workbook

Drop `--dry-run` and add `--export`:

```powershell
uv run python extract_rows.py "../images/score cards with grid/<GAME>.png" `
  --enhance esrgan --reuse-step1 --innings 9 --players 9 `
  --players-file ../players.txt `
  --date YYYY-MM-DD --opponent <NAME> `
  --realign --export --export-out ../stats.xlsx
```

`stats.xlsx` is updated with a new game tab and refreshed season stats.

> **Re-running a game** to fix a mistake: run the same command again. The pipeline detects the existing game, replaces it, and re-exports. Season totals are never double-counted.

---

### Step 5 — Mark reviewed

Once you're satisfied the plays are correct, mark them as reviewed so the team's Low Confidence tab stays clean:

```powershell
uv run python mark_reviewed.py `
  --date YYYY-MM-DD --opponent <NAME> `
  --export --export-out ../stats.xlsx
```

---

## Workbook Structure

| Sheet | Contents |
|---|---|
| Season Stats | Cumulative stats for all players, sorted by OPS |
| Game Log | Per-player per-game line (one row per player per game) |
| 2026-06-07 Almere | Box score for that game (one tab per game) |
| Low Confidence | Plays flagged as uncertain; cleared by `mark_reviewed.py` |

---

## File Layout

```
images/
  score cards with grid/  scans with red divider lines + per-player stats column
  totals_cache/           hand-editable per-inning ground truth
  player_stats_cache/     hand-editable per-player H/AB ground truth

scorecard/
  extract_rows.py       main pipeline
  export_season.py      Excel export
  mark_reviewed.py      mark plays as reviewed in DB
  data/
    season.db           SQLite database (single source of truth)
    raw/                per-game JSON archives
```
