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

## Image Naming Convention

Name scan files as `YYYY-MM-DD_<opponent>.png` and place them in `images/scans/`:

```
images/scans/
  2026-06-07_almere.png
  2026-04-12_quick.png
  2026-06-07_almere_g2.png   ← doubleheader game 2
```

Each scan should include hand-drawn red divider lines between rows **and** the per-player stats column on the right. Add `--no-stats` if a game has no stats column.

---

## Analyzing a New Game

All commands run from the `scorecard/` directory.

### Step 1 — Extract and review

```powershell
uv run python extract_rows.py "../images/scans/YYYY-MM-DD_<opponent>.png" `
  --enhance esrgan --innings 9 --players 9 `
  --players-file ../players.txt `
  --date YYYY-MM-DD --opponent <NAME> `
  --realign --dry-run
```

Read the terminal output. Check:

- **`[LOW]` plays** — cells the model was uncertain about; compare against the physical card
- **Reconciliation mismatches** — `MISSED run(s)`, `EXTRA E#`, etc. at the bottom

If totals were misread (wrong run/hit/error/LOB count), correct the generated ground-truth files and re-run (step 2).

---

### Step 2 — Correct misread totals (only if needed)

Two files are auto-generated in `images/ground_truth/`. Open in any text editor and fix misread numbers:

**`../images/ground_truth/YYYY-MM-DD_<opponent>_totals.txt`** — per-inning team totals from the card's bottom strip:

```
# inning  runs  hits  errors  lob
1    3    1    2    0
2    0    2    0    2
```

**`../images/ground_truth/YYYY-MM-DD_<opponent>_stats.txt`** — per-player H and AB from the stats column:

```
# slot  H  AB
1    3    4
2    0    5
```

You only need to edit these if the numbers don't match the physical card.

---

### Step 3 — Re-run until clean

```powershell
uv run python extract_rows.py "../images/scans/YYYY-MM-DD_<opponent>.png" `
  --enhance esrgan --reuse-step1 --innings 9 --players 9 `
  --players-file ../players.txt `
  --date YYYY-MM-DD --opponent <NAME> `
  --realign --dry-run
```

`--reuse-step1` replays cached vision descriptions — no API calls, instant re-run. Repeat steps 2–3 until the output shows:

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
uv run python extract_rows.py "../images/scans/YYYY-MM-DD_<opponent>.png" `
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
  scans/              ← drop scan images here (YYYY-MM-DD_<opponent>.png)
  ground_truth/       ← auto-generated; hand-correct after each run
    YYYY-MM-DD_<opponent>_totals.txt
    YYYY-MM-DD_<opponent>_stats.txt
  _cache/             ← fully generated; never edit; gitignored
    rows/
    rows_enhanced_esrgan/
    step1/
    stats_crop/

scorecard/
  extract_rows.py     main pipeline
  export_season.py    Excel export
  mark_reviewed.py    mark plays as reviewed in DB
  data/
    season.db         SQLite database (single source of truth)
    raw/              per-game JSON archives
```
