# CLAUDE.md — pipeline architecture & agent notes

Developer/agent reference for the scorecard digitization pipeline. For run instructions see [README.md](README.md).

**Security context (org policy):** subject to CMMC, CRA, PCI, TISAX, EU AI regulations. `scorecard/.env` holds the Gemini API key and is gitignored — never commit API keys or read `.env` contents into chat/output.

---

## High-level flow

```
Scan image (JPG/PNG)
        │
        ▼
1. Grid detection        probe_grid.py / detect_grid()
        │                → row + column boundaries (or manual override, see below)
        ▼
2. Name detection        extract_cells.py / _detect_row_names()
        │                → VLM reads left info strip per row, fuzzy-matched to roster
        │                → substitutions: second name triggers sub_inning prompt
        │                   cached in games/{stem}/cells/_names.json (ALWAYS reused if present,
        │                   independent of --reuse-cache — delete or use --reset-names to redo)
        ▼
3. Per-cell VLM          extract_cells.py / classify_cell()
        │                → one dict per cell: result, run, rbi_slot, confidence
        │                   cached in games/{stem}/cells/r##_c##.json
        ▼
4. RBI backfill          _backfill_rbi_cells()
        │                → focused VLM call on bottom-left quadrant of run=True cells
        │                   to identify which batting slot drove in the run
        │                → only re-reads cells missing the rbi_slot KEY (old-format cache);
        │                   rbi_slot: null is treated as valid cached data
        ▼
5. SB backfill           _backfill_sb_cells()
        │                → focused VLM call counting stolen-base notations in
        │                   top-left/top-right/bottom-left quadrants of reached-base cells
        │                → only re-reads cells missing sb_count key
        ▼
6. Constraint enforcement  _enforce_constraints()
        │                → E (error) is never an out
        │                → out → run=False
        ▼
7. Structural rules      _apply_batting_rules()
        │                → isolation, 3-out, K-PB enforcement
        ▼
8. Hole detection        _reread_hole_cells() / run-reread pass
        │                → null cell sandwiched between two non-null cells re-read
        │                → run=True with result=None re-read; if still no result,
        │                   run is forced to False (grid + cache) so nothing phantom
        │                   flows into stats — cyclic P9→P1 boundary suppressed
        ▼
9. GT enforcement        _enforce_gt_runs()
        │                → fix impossible runs against ground-truth totals
        ▼
10. Reconciliation       main loop
        │                → focused re-check when extracted R < GT R
        ▼
11. Integrity checks     _check_row(), _check_pa_sequence(), _check_col()
        │                → per-player and per-inning cross-checks printed to terminal
        │                → run counts here EXCLUDE cells with result=None even if run=True,
        │                   matching what actually reaches PlateAppearance/stats
        ▼
12. JSON export          GameExtraction (models.py)
        │                → games/{stem}/{stem}_cells.json
        │                → cells with result=None are dropped entirely — never become a PA
        ▼
13. DB write             db.py / write_game()
        │                → season.db (duplicate matched on date+opponent, replaced not double-counted)
        ▼
14. HTML widget          render_widget.py / render_widget_for_game()
           → games/{stem}/{stem}.html
```

---

## Files

| File | Role |
|---|---|
| `extract_cells.py` | Main pipeline entry point. Grid → names → VLM → backfills → constraints → rules → checks → JSON → DB → HTML. |
| `probe_grid.py` | OpenCV grid detector: finds row separators and inning column boundaries; supports manual overrides. |
| `models.py` | Pydantic models: `GameExtraction`, `LineupSlot`, `PlateAppearance`, `InningTotals`, etc. |
| `db.py` | SQLite layer: `init_db`, `write_game`, `find_duplicate_game`, fuzzy player matching. |
| `stats.py` | Derived stat calculations: AVG, OBP, SLG, OPS, BABIP, ISO, wOBA, RC, OPS+, BB/K, AB/HR. SB counted; CS removed from schema. |
| `export_season.py` | Reads DB, writes multi-sheet Excel workbook with color-scaled conditional formatting. No CS column. |
| `reimport.py` | Reimport one `_cells.json` or all games (`--all`) into DB + regenerate HTML. `--sync-cells` reverse-writes a hand-edited `_cells.json` back into the per-cell cache (preserves `rbi_slot`) so `--reuse-cache` doesn't clobber manual fixes. |
| `crawl.py` | Loops game folders under a games dir, re-running `extract_cells.py --reuse-cache --yes` per game (finds the scan in the sibling `scans/` folder). Use to backfill a newly-added field across already-analyzed games. |
| `review.py` | Interactive CLI to review and correct low-confidence PAs. Patches `_cells.json` + DB. |
| `render_widget.py` | Standalone HTML widget renderer (color-coded scorecard grid + per-player stats + SB indicator). |
| `publish.py` | Copies `*.html` (root + up to 2 subfolder levels) and the xlsx to a destination folder; auto-detects the xlsx if not passed. Must be run with `uv run` from `scorecard/`. |
| `mark_reviewed.py` | Bulk-mark PAs as reviewed in the DB. |
| `manage_players.py` | CLI for fuzzy-matched player aliases (confirm, merge, list). |
| `_dump_cells.py` | Debug helper: prints cell cache as CSV (ri, ci, player, result, run, confidence, notes). |

---

## Key design decisions

### Per-cell caching
Each cell result is persisted as `games/{stem}/cells/r{ri:02d}_c{ci:02d}.json` (1-based indices). `--reuse-cache` skips API calls for every cached cell. `api_error` cells always retry.

Cells removed by structural rules are stored as `removed:<rule> (<original_result>)`. On the next run they are **restored** to their original result so rules re-evaluate fresh — making rules stateless and idempotent.

### RBI slot detection
After main classification, `_backfill_rbi_cells()` makes a focused VLM call on the bottom-left quadrant of every `run=True` cell. The quadrant contains either a single batting-order digit 1–9 (the batter who drove in the run) or a multi-character notation (SB/WP/PB/E# = no RBI). `thinking_budget=0` prevents Gemini thinking tokens from consuming the small `max_tokens` budget. On `--reuse-cache`, `rbi_slot: null` is valid cached data; only cells missing the key entirely are backfilled.

### Stolen base (SB) detection
`_backfill_sb_cells()` targets reached-base, non-out cells missing `sb_count`. Counts SB notations in the top-left/top-right/bottom-left quadrants (never bottom-right, which is RBI/out territory). `PlateAppearance.sb = int(cell.get("sb_count") or 0)`. Flows through `stats.py SB` → `export_season.py` (gold-highlighted season leader column, included in per-game and game-log sheets). CS was removed from the schema/exports — SB only.

### Voided / misscored cells
The VLM prompt instructs: if the bottom-right quadrant alone is crossed out with diagonal lines, or all four quadrants are crossed out, the PA was scored in the wrong cell — return `result: null` rather than guessing.

### Constraint enforcement
Applied before structural rules, in order:
1. **E (error) rule** — any result matching `^E\d*` is never an out.
2. **Out-run rule** — if a cell is an out (and not an error), `run` is forced to False.

### Structural rules (batting rules)
1. **Isolation rule** — a PA with neither predecessor nor successor in the same inning is removed unless a neighbor is uncertain.
2. **3-out rule** — once 3 outs are recorded in batting order within an inning, all subsequent PAs in that inning are removed. Skipped for innings with uncertain cells.

`K-PB` (dropped third strike) is **not** an out for rules 1 and 3.

### Hole detection & the null-result/run=True consistency rule
A null cell sandwiched between two non-null cells in the same column is re-read (cyclic P9→P1 boundary suppressed — that's end-of-inning, not a hole). Separately, `_reread_hole_cells`-adjacent logic re-reads any `run=True, result=None` cell; **if the re-read still fails to produce a result, `run` is explicitly forced to `False`** in both the in-memory grid and the cache file. This matters because `PlateAppearance` construction (`extract_cells.py`, in the per-row PA-building loop) skips any cell with `result is None` entirely — such a cell contributes nothing to stats or the DB regardless of its `run` flag. Before this fix, a lingering `run=True/result=None` cell could make the per-inning integrity check report a phantom extra run that would never actually appear in the final output — always keep `run` and `result` consistent when hand-patching cells for this reason.

### Ground-truth enforcement
`_enforce_gt_runs()` runs after batting rules. For any inning where GT says R=0, all `run=True` cells are forced False. Over-counted innings are logged but not auto-corrected (needs a human to pick which cell is wrong).

### PA cross-check identity
For a completed inning: `PA = 3 (outs) + R (runs) + LOB`. Checked in `_check_col()` using ground-truth totals. `_check_col` and the run-reconciliation loop both only count a cell as a "run" if it has `run=True` **and** a non-null `result` — matching the final PlateAppearance filter (see above).

### Player name and substitution detection
The left info strip of each row is cropped and sent to the VLM. The active player for a given cell is resolved from `_names.json` using `sub_innings`: the first name is the starter; subsequent names apply from the listed inning onwards. A detected sub-slot name that's too short after stripping dots/spaces (< 4 significant chars) and doesn't fuzzy-match the roster is treated as VLM noise and silently dropped rather than prompted — real abbreviated names are at least "A. X" (4 chars).

`players.txt` is appended to when a new player is chosen interactively; the append now guards against a missing trailing newline on the existing file (a prior bug concatenated a new name onto the last line, then repeated appends across sessions produced a badly duplicated roster — always eyeball `players.txt` after a bulk name-correction session).

### Grid detection (probe_grid.py)
V-line and H-line detection use OpenCV projection + gap analysis, with a bimodal check (full-row vs sub-row divider gaps in ~2:1 ratio). The bimodal cell-size is computed as `lower_med * 2` (sub-row divider height × 2), **not** `upper_med` — the upper cluster can undershoot when some full-row gaps are noisy or partially missed. Outlier full-row gaps (>3× median, from missed H-lines) fall back to the median instead of dominating the estimate.

For difficult scans (low-res, landscape, skewed, wide player-info columns), manual overrides bypass detection:
- `left_skip_frac` / `--left-skip` — fraction of image width to skip before V-line detection (increase for wide player-info areas).
- `grid_start` + `grid_col_width` / `--grid-start` + `--grid-width` — force a uniform column grid, skipping V-line detection entirely.
- `cell_height` / `--cell-height` — override the detected row height directly when bimodal detection still gets it wrong.
- Left column extrapolation: if fewer inning columns are detected than expected, missing columns are synthesized by stepping backwards from the first detected column (not forwards from an assumed start — that produced gap mismatches).

Example real override that was needed for a 1345×948px landscape/low-res scan (2026-28-06 Vennep Flyers): `--innings 10 --grid-start 455 --grid-width 75 --left-skip 0.35`.

### VLM model
Default: `gemini-2.5-flash` (set via `EXTRACTION_MODEL` env var or `--model`). The prompt describes all valid result codes, K-PB detection, voided-cell handling, and sub-cell notation conventions (WP/PB/SB = baserunner advancement, not PA result).

### Color scale (Excel)
Conditional formatting uses data-relative `min`/`max` anchors so the full blue-to-red gradient always spans the actual player range. White = team average (computed from players with significant PA count).

### AB calculation
BB, HBP (both `HP` and `HBP`), SAC/SH, and SF do not count as an at-bat. `AB = PA − BB − HP − SAC − SF` in `stats.py`; the same exclusion applies in the game log sheet of the Excel export.

---

## DB schema (key tables)

| Table | Key columns |
|---|---|
| `games` | `game_id`, `date`, `opponent`, `game_number`, `raw_json_path` |
| `players` | `player_id`, `name` |
| `plate_appearances` | `pa_id`, `player_id`, `game_id`, `inning`, `batting_order`, `result`, `run_scored`, `rbi`, `sb`, `bb`, `hp`, `sac`, `sf`, `confidence`, `needs_review` |

Duplicate detection: on reimport, any existing game with the same date + opponent is deleted before re-inserting. Fuzzy player matching uses `fuzzy.auto_match_threshold` in `config.yml`.

---

## Known-stale things to watch for

- The project's saved auto-memory (outside this repo) still references an older `process_game.py` / `extract_rows.py` / Anthropic-vision architecture from an earlier iteration of this project — that pipeline no longer exists. Trust this file and the code over old memory notes if they conflict.
- `players.txt` should have one player per line, no duplicates, trailing newline present — check it after any bulk name-correction session.
