# KNBSB Scorecard Pipeline

Digitizes Dutch KNBSB baseball scorecards into structured JSON, a SQLite database, an Excel workbook, and a color-coded HTML widget — using Gemini 2.5 Flash as the VLM backbone.

For pipeline architecture and internals, see [CLAUDE.md](CLAUDE.md).

## Usage

Run every command from the `scorecard/` folder, through `uv run` (it uses the project's environment; `.\scorecard.py` on its own just opens the file):

```powershell
cd C:\Users\vsp\Documents\ScorecardApp\scorecard
uv run python scorecard.py --help                  # list all commands
uv run python scorecard.py <command> --help        # options of one command
```

**A typical game:**
```powershell
# 1. put the scan in "Quick 2026/scans/" as "YYYY-MM-DD - Opponent (Home|Away).jpg"
uv run python scorecard.py read-scorecard "2026-09-27 Herons (Home)" --innings 8
# 2. look at the widget (.html) and the terminal checks, then fix flagged plays
uv run python scorecard.py review-cells "2026-09-27 Herons (Home)"
# 3. refresh the workbook
uv run python scorecard.py export-excel
```

Or use the GUI: `uv run streamlit run app.py` (see [GUI](#gui)).

**GAME** — every per-game command accepts the full game name (`"2026-09-27 Herons (Home)"`), the scan, the game folder, or any file in it. `review-cells` and `mark-reviewed` also take a fragment such as the date.

**Season folder** — commands target `paths.data_root` in `scorecard/config.yml` (now `../Quick 2026`). For another season pass `--data-root "Quick 2026 - Ex Spring Training"` where offered, or set it for the whole PowerShell session with `$env:SCORECARD_DATA_ROOT = "Quick 2026 - Ex Spring Training"`. A relative folder is looked up from where you run the command, then the project folder, then `scorecard/`; a folder that doesn't exist is an error. See [Multiple teams / seasons](#multiple-teams--seasons).

## Commands

| Command | What it does |
|---|---|
| [`read-scorecard`](#read-scorecard) | Read a scan into stats: grid, cell reading, checks, DB, widget |
| [`review-cells`](#review-cells) | Walk through low-confidence plate appearances and correct them |
| [`reimport-game`](#reimport-game) | Load `_cells.json` into the DB and redraw the widget (one game or `--all`) |
| [`sync-edits`](#sync-edits) | Copy hand edits in `_cells.json` back into the cell cache |
| [`reread-season`](#reread-season) | Re-run `read-scorecard` from cache for every game |
| [`fix-date`](#fix-date) | Correct a game's date (folder, files, JSON, DB) |
| [`mark-reviewed`](#mark-reviewed) | Mark all plate appearances of a game as reviewed |
| [`players`](#players) | Player names: list, aliases, merge |
| [`export-excel`](#export-excel) | Write the season Excel workbook |
| [`publish`](#publish) | Copy widgets + workbook to a folder |
| [`draw-widget`](#draw-widget) | Redraw a game's HTML widget |
| [`test-accuracy`](#test-accuracy) | Measure cell-reading accuracy (API calls) |
| [`run-tests`](#run-tests) | Run all offline tests |

The old script names (`extract_cells.py`, `review.py`, `reimport.py`, `crawl.py`, `correct_date.py`, `mark_reviewed.py`, `manage_players.py`, `export_season.py`, `publish.py`, `render_widget.py`, `eval_cells.py`) still work with the same options.

### read-scorecard
`read-scorecard [OPTIONS] GAME` — read a scanned scorecard (old: `extract_cells.py`).

| Option | Meaning |
|---|---|
| `--innings N` | Innings played. Default 9, or the value a prior run of this game stored in `cells/_layout.json` |
| `--active-players N` | Active batting slots (subs share a slot). Default 9 |
| `--n-player-rows N` | Physical player rows on the card (template size, usually 10) |
| `--reuse-cache` | Reuse cached per-cell reads (no API calls for cached cells) |
| `-y`, `--yes` | Skip all interactive prompts (names, totals review) |
| `--dry-run` | Read and check, but don't write to the DB |
| `--reset-names` | Re-detect player names from scratch (combine with `--reuse-cache` to keep cell reads) |
| `--players PATH` | Roster file (default: `players.txt` in the season folder) |
| `--model NAME` | VLM model (default: `EXTRACTION_MODEL` in `.env`, e.g. `gemini-2.5-flash`) |
| `--workers N` | Parallel VLM calls. Default 8 |
| `--verify-model NAME` | Model for the second read of uncertain cells (default: `--model`) |
| `--no-verify` | Skip the second read of uncertain cells |
| `--read-player-stats` | Read per-player H-AB from card columns 11–12 into `_stats.txt` |
| `--gt-dir PATH` | Ground-truth folder (default: the game folder) |
| `--legacy-grid` | Use the old grid detector instead of the lattice fit |
| `--left-skip F` | Legacy grid: fraction of image width to skip (wide player-info area) |
| `--grid-start X` + `--grid-width W` | Legacy grid: force a uniform column grid |
| `--cell-height H` | Legacy grid: override the detected row height |

Grid flags and `--innings`/`--n-player-rows` only need to be given once; see [Grid detection overrides](#grid-detection-overrides).

### review-cells
`review-cells [OPTIONS] [GAME]` — interactive review (old: `review.py`). Without GAME: the flags of every game.

| Option | Meaning |
|---|---|
| `--all` | Every PA of the game, not just low-confidence flags (needs GAME) |
| `--max-conf N` | Any unreviewed PA with confidence ≤ N (1–5) |
| `--game-id N` | Select the game by DB id |
| `--game GAME` | Same as GAME (old form) |
| `--db PATH` | Use another DB file |

At each prompt: Enter = correct as-is · `1B` / `K` / `BB` … = new result · `y` / `n` = run scored or not · `1B y` = both · `d` = delete PA · `q` = quit. Every PA you correct or confirm becomes hand-checked (confidence 5/5) in the DB, `_cells.json` and the cell cache, so later `--reuse-cache` runs keep it.

### reimport-game
`reimport-game [OPTIONS] [GAME]` — load `_cells.json` into the DB and redraw the widget (old: `reimport.py`). No API calls, no pipeline rules.

| Option | Meaning |
|---|---|
| `--all` | Every game of the season (GAME may then be a games folder). Games whose folder was deleted are removed from the DB |
| `--sync-cells` | Write `_cells.json` into the cell cache instead (same as `sync-edits`) |
| `--data-root FOLDER` | Season folder |

### sync-edits
`sync-edits [OPTIONS] GAME` — copy hand edits in `_cells.json` back into the cell cache, so `--reuse-cache` / `reread-season` don't overwrite them. Option: `--data-root FOLDER`.

### reread-season
`reread-season [OPTIONS] [GAMES_DIR]` — run `read-scorecard --reuse-cache --yes` for every game (old: `crawl.py`). Re-applies all pipeline rules and checks; new or failed cells cost API calls. Rebuilds `_cells.json` from the cell cache, so run `sync-edits` first for hand edits made only in the JSON.

| Option | Meaning |
|---|---|
| `--game TEXT` | Only games whose folder name contains TEXT (a date or the full name) |
| `--data-root FOLDER` | Season folder |

### fix-date
`fix-date [OPTIONS] GAME NEW_DATE` — correct a mistyped date (old: `correct_date.py`). Renames the game folder, its files and the scan, patches the JSON and the DB, redraws the widget. Date only (not opponent or home/away).

| Option | Meaning |
|---|---|
| `--dry-run` | Show what would change, do nothing |
| `--data-root FOLDER` | Season folder |

### mark-reviewed
`mark-reviewed [OPTIONS] [GAME]` — mark all PAs of a game as reviewed (old: `mark_reviewed.py`).

| Option | Meaning |
|---|---|
| `--export` | Re-export the workbook afterwards |
| `--export-out PATH` | Workbook path for `--export` (default `../stats.xlsx`) |
| `--date D --opponent O` | Old way to select the game; `--game-number N` for a doubleheader |

### players
`players COMMAND` — player names (old: `manage_players.py`).

| Command | Meaning |
|---|---|
| `players list` | All players and their aliases |
| `players aliases` | Review pending fuzzy matches from the last import |
| `players merge NAME1 NAME2` | Merge NAME2 into NAME1 (moves all PAs, adds an alias) |

### export-excel
`export-excel [OPTIONS]` — write the season workbook from the DB (old: `export_season.py`). See [Excel workbook](#excel-workbook).

| Option | Meaning |
|---|---|
| `--data-root FOLDER` | Season folder |
| `--output PATH` | Workbook path (default: `<season>/<season> stats.xlsx`) |
| `--min-pa N` | Only players with at least N PAs |

### publish
`publish [OPTIONS] [SOURCE] DEST` — copy all HTML widgets and the workbook to DEST, e.g. a Google Drive folder (old: `publish.py`).

| Option | Meaning |
|---|---|
| `--data-root FOLDER` | Season folder to use as SOURCE (SOURCE can then be left out) |
| `--xlsx PATH` | Workbook to copy (auto-detected if omitted) |

```powershell
uv run python scorecard.py publish "../Quick 2026" "G:\My Drive\Quick 2026"
```

### draw-widget
`draw-widget [OPTIONS] GAME` — redraw a game's HTML widget from its `_cells.json` (old: `render_widget.py`). Option: `--no-open` (don't open it in the browser).

### test-accuracy
`test-accuracy [OPTIONS]` — re-read every cell of the hand-verified games (Grizzlies 2026-07-12, Urbanus 2026-08-23) and score the reads against them (old: `eval_cells.py`). Run before and after a prompt change; results vary by 2–3 cells between runs.

| Option | Meaning |
|---|---|
| `--check` | No API calls: label stats and cache vs `_cells.json` consistency |
| `--only all\|cell\|rbi` | Score only cell reads or only RBI digits |
| `--model NAME` | Model to test (default: `EXTRACTION_MODEL`) |
| `--workers N` | Parallel calls |
| `--thinking N` | Override the Gemini thinking budget |
| `--save-crops` | Save crops of misread cells to `eval_out/<run>/` |

### run-tests
`run-tests` — all offline tests (no API calls), including `test-accuracy --check`.

---

## Setup

```powershell
cd scorecard
uv sync
```

Create `scorecard/.env` with your Gemini API key (gitignored — never commit it):

```
GOOGLE_API_KEY=AIza...
EXTRACTION_MODEL=gemini-2.5-flash
```

## Directory layout

```
Quick 2026/
  games/
    2026-04-12 - Thamen (Home)/
      cells/                    ← per-cell VLM cache (TRACKED in git: holds hand-corrected cells)
      2026-04-12 - Thamen (Home)_cells.json   ← main output; edit to fix errors
      2026-04-12 - Thamen (Home)_grid_debug.png
      2026-04-12 - Thamen (Home).html         ← color-coded visual widget
      2026-04-12 - Thamen (Home)_rectified.jpg ← straightened photo (only when the image was skewed)
      2026-04-12 - Thamen (Home)_totals.txt    ← per-inning R/H/E/LOB (auto-read from the card; edit to fix)
      2026-04-12 - Thamen (Home)_stats.txt     ← optional per-slot H/AB (see --read-player-stats)
  scans/                        ← drop scan images here
  players.txt                   ← team roster (name, jersey per line)
  Quick 2026.db                 ← SQLite database, named after the season folder (gitignored)
  Quick 2026 stats.xlsx         ← generated by export-excel (gitignored)

scorecard/
  scorecard.py         ← single entry point for every command
  app.py               ← Streamlit GUI
  extract_cells.py     ← main pipeline (read-scorecard)
  rectify.py           ← photo straightening + grid lattice fit (default grid detector)
  probe_grid.py        ← legacy OpenCV grid detection (manual overrides / fallback)
  gamepaths.py         ← resolves GAME arguments (folder, file, scan or name)
  eval_cells.py        ← test-accuracy
  reimport.py          ← reimport-game / sync-edits
  crawl.py             ← reread-season
  export_season.py     ← export-excel
  review.py            ← review-cells
  publish.py           ← publish
  mark_reviewed.py     ← mark-reviewed
  correct_date.py      ← fix-date
  manage_players.py    ← players
  render_widget.py     ← draw-widget
  stats.py             ← derived stat calculations (AVG, OBP, SLG, wOBA, OPS+, …)
  db.py, models.py, config.yml
```

## Image naming

Name scan files as `YYYY-MM-DD - <Opponent> (Home|Away).jpg` and place them in `Quick 2026/scans/`. Date, opponent, and home/away are parsed from the filename. Outputs land under `Quick 2026/games/{stem}/`.

## GUI

```powershell
cd scorecard
uv run streamlit run app.py
```

Opens `http://localhost:8501` (local only; usage telemetry off, see `scorecard/.streamlit/config.toml`).
- **Games:** all games and how many plate appearances need review; add a scan (name it `YYYY-MM-DD - Opponent (Home|Away)`); **Read scorecard** with a live log; the widget and the grid debug image.
- **Review:** every low-confidence plate appearance with the image of its cell, the reading, and why it was flagged. Change the result/run if needed and press **Save**: it's written to `_cells.json`, synced into the cell cache and the DB immediately, and marked hand-checked (full confidence, never re-flagged).
- **Season:** export the Excel workbook, publish.

## Processing a game

### 1 — Run the extraction

```powershell
uv run python scorecard.py read-scorecard "2026-06-07 - Almere (Away)"
```

The pipeline detects the grid, reads player names/subs, classifies every PA cell via the VLM, enforces logical/structural rules, cross-checks against ground truth, then writes `_cells.json` / `_grid_debug.png` / `.html` and imports into the season DB.

### 2 — Check the output

Open the generated `.html` widget. Color coding: **green** = hit, **blue** = reached base without a hit (BB/HBP), **yellow** = error/FC/K-PB, **red** = out. Also check the terminal: in the per-player and per-inning check tables, any stat that disagrees with ground truth is annotated inline, e.g. `R=4 (GT=5)`; stats that match print bare.

### 3 — (Optional) supply ground truth

**`{game}_totals.txt`** — per-inning team totals. Read automatically from the card's "Totaal per inning" row on the first run and written here; edit it if a number was misread, then re-run with `--reuse-cache`:
```
# inning  runs  hits  errors  lob
1  3  1  2  0
```

**`{game}_stats.txt`** — per batting slot H and AB, the slot's total over starter + subs. Not read by default (columns 11–12 are often empty and include sac flies, which aren't tracked yet); `--read-player-stats` reads them from the card, keeping only lines whose H-AB matches the written average:
```
# slot  H  AB
1  3  4
5  2  5
```

### 4 — Re-run with cache

```powershell
uv run python scorecard.py read-scorecard "2026-06-07 - Almere (Away)" --reuse-cache
```

`--reuse-cache` skips API calls for cells already classified; structural rules always re-run. Repeat 2–4 until checks pass.

### Photos and what the pipeline checks for you

Phone photos work: a skewed or perspective photo is straightened automatically (saved as `{game}_rectified.jpg`) and the grid is fitted to the printed lines, so no grid flags are needed. Check `{game}_grid_debug.png` after the first run.

While reading, the pipeline cross-checks itself and lowers a plate appearance's confidence (so `review-cells` shows it) when something doesn't add up:
- empty cells (no pen ink) are skipped without an API call;
- a large circle means an out: if the image and the reading disagree, the cell is re-read;
- ink in the bottom-left quadrant means a run: a missed run is re-checked;
- written notations in the base quadrants mean the batter reached base: a "no PA" read is re-checked;
- an inning that ran over into the next column is detected from the blank totals cell under it; in such an inning a misread 3rd out no longer deletes the following PAs;
- uncertain cells get a second, independent read (`--verify-model` to use another model, `--no-verify` to skip); a disagreement sends the cell to review;
- inning totals (R/H) and, if present, per-slot H/AB must match the card.

To measure reading accuracy after changing prompts: `scorecard.py test-accuracy` (uses the hand-verified games as answer key).

### Grid detection overrides

For low-res, landscape or oddly laid-out scans: if grid auto-detection produces a wrong column/row count or spacing, check the debug image (`*_grid_debug.png`) and override manually:

| Flag | Use for |
|---|---|
| `--left-skip 0.35` | Wide player-info area throwing off V-line detection (increase from default 0.05) |
| `--grid-start 455 --grid-width 75` | Bypass V-line detection entirely; force a uniform column grid |
| `--cell-height 74` | Bimodal H-line detection picks the wrong row height (e.g. sub-row divider height instead of full row) |
| `--reset-names` | Delete the cached player names and re-detect from scratch (use with `--reuse-cache` to keep cell reads) |

You only need to pass these (and `--innings`/`--n-player-rows`) once — they're persisted in `cells/_layout.json` and reused automatically on a later `--reuse-cache` run of the same game that omits them, so `reread-season` still gets the right grid for a game that needed an override.

## Correcting a PA after extraction

**Interactive reviewer** (recommended for low-confidence flags):
```powershell
uv run python scorecard.py review-cells "2026-06-07 - Almere (Away)"           # only low-confidence flags
uv run python scorecard.py review-cells 2026-06-07 --max-conf 3                # widen to confidence <= 3
uv run python scorecard.py review-cells 2026-06-07 --all                       # every PA in the game
```
Each PA is tagged with its confidence (1-5) and why it was docked, e.g. `[conf 2/5: reread:hole; inning_R_mismatch]`.

**Manual edit:** edit `Quick 2026/games/{stem}/{stem}_cells.json` directly, then:
```powershell
uv run python scorecard.py reimport-game "2026-04-12 - Thamen (Home)"
uv run python scorecard.py sync-edits "2026-04-12 - Thamen (Home)"
```
Without `sync-edits` the per-cell cache (`cells/r##_c##.json`) still holds the old values and overwrites your edit on a future `--reuse-cache` run or `reread-season`.

## Fixing a mistyped game date

If a game folder was named with the wrong date (e.g. month/day swapped), don't re-run extraction — `fix-date` renames the game folder, its `_cells.json`/`.html`/log/debug-image files, and the matching scan in `scans/`, then patches `game.date` in the JSON and (if already imported) the DB `games` row, and regenerates the widget. The per-cell cache in `cells/` moves along with the folder.
```powershell
uv run python scorecard.py fix-date "2026-04-12 - Thamen (Home)" 2026-12-04
uv run python scorecard.py fix-date --dry-run "2026-04-12 - Thamen (Home)" 2026-12-04   # preview only
```
Then run `export-excel` to refresh the workbook, since game dates changed.

## Bulk operations

**`reimport-game --all` vs `reread-season`:**
- `reimport-game --all` loads every game's `_cells.json` as-is into the DB and redraws the widgets. Seconds, no API calls. Use it when the DB is missing or out of date (also after deleting a game folder: the game is removed from the DB).
- `reread-season` re-runs the whole pipeline from the scan + cell cache for every game, re-applying all rules and checks. Minutes, some API calls. Use it after a pipeline change.

```powershell
uv run python scorecard.py reimport-game --all
uv run python scorecard.py reread-season
uv run python scorecard.py reread-season --game "2026-04-12"   # single game by date fragment
```

**Removing a game:** delete (or move) its folder under `games/`, then run `reimport-game --all`.

## Excel workbook

```powershell
uv run python scorecard.py export-excel
```

Output: `Quick 2026/Quick 2026 stats.xlsx`

| Sheet | Contents |
|---|---|
| Season Stats | Cumulative stats for all players, sorted by OPS |
| Game Log | Per-player per-game line |
| {Date} {Opponent} | Box score for each game |
| Low Confidence | PAs with confidence <= `review.threshold` (config.yml, default 2), with the reasons they were docked |

Stats: PA, AB, H, 2B, 3B, HR, BB, K, R, RBI, SB, AVG, OBP, SLG, OPS, BABIP, ISO, BB%, K%, wOBA, RC, OPS+, AB/HR, BB/K. BB, HBP, SAC, and SF do not count as an AB.

## Multiple teams / seasons

By default every command targets the season folder in `scorecard/config.yml` (`paths.data_root`, now `"../Quick 2026"`). To target another folder (another team, or spring training games kept apart from the regular season) without editing `config.yml`, pass `--data-root` to `export-excel`, `reimport-game`, `sync-edits`, `reread-season`, `fix-date` or `publish`:

```powershell
uv run python scorecard.py reimport-game --all --data-root "Quick 2026 - Ex Spring Training"
uv run python scorecard.py export-excel --data-root "Quick 2026 - Ex Spring Training"
uv run python scorecard.py reread-season --data-root "Quick 2026 - Ex Spring Training"
uv run python scorecard.py publish --data-root "Quick 2026 - Ex Spring Training" "G:\My Drive\Quick 2026 - Ex Spring Training"
```

Or set it once per PowerShell session so every command after it (including `read-scorecard` and `review-cells`) uses that folder:
```powershell
$env:SCORECARD_DATA_ROOT = "Quick 2026 - Ex Spring Training"
uv run python scorecard.py export-excel
```

A relative folder is looked up from where you run the command, then the project folder, then `scorecard/`; a folder that exists in none of them is an error, never a new empty season. Each season folder has its own SQLite DB (named after the folder, e.g. `Quick 2026 - Ex Spring Training.db`) and its own `players.txt` — rosters are not shared between seasons/teams.

## Roster

Player names are detected from the scorecard automatically and fuzzy-matched against `Quick 2026/players.txt`. Substitutions share the same batting slot and are detected from the info strip.

```powershell
uv run python scorecard.py players list
uv run python scorecard.py players aliases    # audit recent fuzzy matches
```

## Notes

- **API key safety**: `.env` is gitignored. Never commit API keys. (Org policy: CMMC, CRA, PCI, TISAX, EU AI regulations apply.)
- **Re-running a game**: the pipeline replaces any existing DB entry automatically (matched on date + opponent).
- **DB path**: driven by `paths.data_root` in `scorecard/config.yml`, overridable per-command with `--data-root` or the `SCORECARD_DATA_ROOT` env var (see [Multiple teams / seasons](#multiple-teams--seasons)).
