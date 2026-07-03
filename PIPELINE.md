# Pipeline architecture reference

Developer reference for the scorecard digitization pipeline. For run instructions see [README.md](README.md).

---

## High-level flow

```
Scan image (JPG/PNG)
        │
        ▼
1. Grid detection        probe_grid.py / detect_grid()
        │                → row + column boundaries
        ▼
2. Name detection        extract_cells.py / _detect_row_names()
        │                → VLM reads left info strip per row
        │                → fuzzy-matched to roster
        │                → substitutions: second name triggers sub_inning prompt
        │                   cached in games/{stem}/cells/_names.json
        ▼
3. Per-cell VLM          extract_cells.py / classify_cell()
        │                → up to 81 raw cell dicts (result, run, rbi_slot, confidence)
        │                   cached in games/{stem}/cells/r##_c##.json
        ▼
4. RBI backfill          _backfill_rbi_cells()
        │                → focused VLM call on bottom-left quadrant of run=True cells
        │                   to identify which batting slot drove in the run
        ▼
5. Constraint enforcement  _enforce_constraints()
        │                → E (error) is never an out
        │                → out → run=False
        ▼
6. Structural rules      _apply_batting_rules()
        │                → isolation, 3-out, K-PB enforcement
        ▼
7. Hole detection        _reread_hole_cells()
        │                → null cell sandwiched between two non-null cells re-read
        │                   cyclic P9→P1 boundary suppressed (end-of-inning, not a hole)
        ▼
8. GT enforcement        _enforce_gt_runs()
        │                → fix impossible runs against ground-truth totals
        ▼
9. Reconciliation        main loop
        │                → focused re-check when extracted R < GT R
        ▼
10. Integrity checks     _check_row(), _check_pa_sequence(), _check_col()
        │                → per-player and per-inning cross-checks printed to terminal
        ▼
11. JSON export          GameExtraction (models.py)
        │                → games/{stem}/{stem}_cells.json
        ▼
12. DB write             db.py / write_game()
        │                → season.db  (duplicate replaced, never double-counted)
        ▼
13. HTML widget          render_widget.py / render_widget_for_game()
           → games/{stem}/{stem}.html
```

---

## Files

| File | Role |
|---|---|
| `extract_cells.py` | Main pipeline entry point. Grid → VLM → constraints → rules → checks → JSON → DB → HTML. |
| `probe_grid.py` | OpenCV grid detector: finds row separators and inning column boundaries. |
| `models.py` | Pydantic models: `GameExtraction`, `LineupSlot`, `PlateAppearance`, `InningTotals`, etc. |
| `db.py` | SQLite layer: `init_db`, `write_game`, `find_duplicate_game`, fuzzy player matching. |
| `stats.py` | Derived stat calculations: AVG, OBP, SLG, OPS, BABIP, ISO, wOBA, RC, OPS+, BB/K, AB/HR. |
| `export_season.py` | Reads DB, writes multi-sheet Excel workbook with color-scaled conditional formatting. |
| `reimport.py` | Reimport one `_cells.json` or all games under a directory into DB + regenerate HTML. |
| `review.py` | Interactive CLI to review and correct low-confidence PAs. Patches `_cells.json` + DB. |
| `render_widget.py` | Standalone HTML widget renderer (color-coded scorecard grid + per-player stats). |
| `publish.py` | Copy game HTML widgets and stats xlsx to a destination folder (e.g. Google Drive). |
| `mark_reviewed.py` | Bulk-mark PAs as reviewed in the DB. |
| `manage_players.py` | CLI for fuzzy-matched player aliases (confirm, merge, list). |
| `_dump_cells.py` | Debug helper: prints cell cache as CSV (ri, ci, player, result, run, confidence, notes). |

---

## Key design decisions

### Per-cell caching
Each cell result is persisted as `games/{stem}/cells/r{ri:02d}_c{ci:02d}.json` (1-based indices). `--reuse-cache` skips API calls for every cached cell. `api_error` cells always retry.

Cells removed by structural rules are stored as `removed:<rule> (<original_result>)`. On the next run they are **restored** to their original result so rules re-evaluate fresh — making rules stateless and idempotent.

On `--reuse-cache`, cells with `rbi_slot: null` are treated as valid cached data and not re-sent to the API. Only cells missing the `rbi_slot` key entirely (old-format cache) are backfilled.

### RBI slot detection
After main classification, `_backfill_rbi_cells()` makes a focused VLM call on the bottom-left quadrant of every `run=True` cell. The quadrant contains either a single batting-order digit 1–9 (the batter who drove in the run) or a multi-character notation (SB/WP/PB/E# = no RBI). `thinking_budget=0` is set to prevent Gemini thinking tokens from consuming the small `max_tokens` budget.

### Constraint enforcement
Applied before structural rules, in order:
1. **E (error) rule** — any result matching `^E\d*` is never an out.
2. **Out-run rule** — if a cell is an out (and not an error), `run` is forced to False.

### Structural rules (batting rules)
1. **Isolation rule** — a PA with neither predecessor nor successor in the same inning is removed unless a neighbor is uncertain.
2. **3-out rule** — once 3 outs are recorded in batting order within an inning, all subsequent PAs in that inning are removed. Skipped for innings with uncertain cells.

`K-PB` (dropped third strike) is **not** an out for rules 1 and 3.

### Hole detection
A null cell sandwiched between two non-null cells in the same column is re-read. The cyclic batting order (P9 → P1) is handled: if an inning starts at P1 (`min_nonnull == 0`), a null at P9 is end-of-inning, not a hole, and is suppressed.

### AB calculation
BB, HBP (both `HP` and `HBP`), SAC/SH, and SF do not count as an at-bat. `AB = PA − BB − HP − SAC − SF` in `stats.py`; the same exclusion applies in the game log sheet of the Excel export.

### Ground-truth enforcement
`_enforce_gt_runs()` runs after batting rules. For any inning where GT says R=0, all `run=True` cells are forced False. Over-counted innings are logged but not auto-corrected.

### PA cross-check identity
For a completed inning: `PA = 3 (outs) + R (runs) + LOB`. Checked in `_check_col()` using ground-truth totals.

### Player name and substitution detection
The left info strip of each row is cropped and sent to the VLM. The active player for a given cell is resolved from `_names.json` using `sub_innings`: the first name is the starter; subsequent names apply from the listed inning onwards.

### VLM model
Default: `gemini-2.5-flash` (set via `EXTRACTION_MODEL` env var or `--model`). The prompt describes all valid result codes, K-PB detection, and sub-cell notation conventions (WP/PB/SB = baserunner advancement, not PA result).

### Color scale (Excel)
Conditional formatting uses data-relative `min`/`max` anchors so the full blue-to-red gradient always spans the actual player range. White = team average (computed from players with significant PA count).

---

## DB schema (key tables)

| Table | Key columns |
|---|---|
| `games` | `game_id`, `date`, `opponent`, `game_number`, `raw_json_path` |
| `players` | `player_id`, `name` |
| `plate_appearances` | `pa_id`, `player_id`, `game_id`, `inning`, `batting_order`, `result`, `run_scored`, `rbi`, `bb`, `hp`, `sac`, `sf`, `confidence`, `needs_review` |

Duplicate detection: on reimport, any existing game with the same date + opponent is deleted before re-inserting. Fuzzy player matching uses `auto_match_threshold: 70` (configurable in `config.yml`).
