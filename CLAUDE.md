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
        │                → one dict per cell: result, run, rbi_slot (run=True cells only,
        │                   see #7 below), result_conf, run_conf
        │                   cached in games/{stem}/cells/r##_c##.json
        ▼
4. SB backfill           _backfill_sb_cells()
        │                → focused VLM call counting stolen-base notations in
        │                   top-left/top-right/bottom-left quadrants of reached-base cells
        │                → only re-reads cells missing sb_count key
        ▼
5. Constraint enforcement  _enforce_constraints()
        │                → E (error) is never an out
        │                → out → run=False
        │                → HR → run=True
        ▼
6. Structural rules      _apply_batting_rules()
        │                → isolation, 3-out, K-PB enforcement
        ▼
7. Hole detection        _reread_hole_cells() / run-reread pass
        │                → null cell sandwiched between two non-null cells re-read
        │                → run=True with result=None re-read; if still no result,
        │                   run is forced to False (grid + cache) so nothing phantom
        │                   flows into stats — cyclic P9→P1 boundary suppressed
        ▼
8. GT enforcement        _enforce_gt_runs()
        │                → fix impossible runs against ground-truth totals
        ▼
9. Reconciliation        main loop
        │                → focused re-check when extracted R < GT R
        ▼
10. Integrity checks     _check_row(), _check_pa_sequence(), _check_col()
        │                → per-player and per-inning cross-checks printed to terminal
        │                → run counts here EXCLUDE cells with result=None even if run=True,
        │                   matching what actually reaches PlateAppearance/stats
        ▼
11. RBI backfill         _backfill_rbi_cells()
        │                → focused VLM call on bottom-left quadrant of run=True cells
        │                   missing an rbi_slot key, to identify which batting slot drove
        │                   in the run — rbi_slot: null (key present) is valid cached data
        │                → runs LAST (#7): any of steps 5-9 above can flip a cell's run
        │                   False→True, and rbi_slot is only meaningful once run has settled
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
| `reimport.py` | Reimport one `_cells.json` or all games (`--all`) into DB + regenerate HTML. `--sync-cells` reverse-writes a hand-edited `_cells.json` back into the per-cell cache (preserves `rbi_slot`, `reread`, `adjusted`; writes back `result_conf`/`run_conf` if present) so `--reuse-cache` doesn't clobber manual fixes. |
| `crawl.py` | Loops game folders under a games dir, re-running `extract_cells.py --reuse-cache --yes` per game (finds the scan in the sibling `scans/` folder). Use to backfill a newly-added field across already-analyzed games. |
| `review.py` | Interactive CLI to review and correct low-confidence PAs (default: `needs_review=1`; `--max-conf N` widens to any unreviewed PA with confidence <= N; `--all` shows every PA). Patches `_cells.json` + DB. |
| `render_widget.py` | Standalone HTML widget renderer (color-coded scorecard grid + per-player stats + SB indicator). |
| `publish.py` | Copies `*.html` (root + up to 2 subfolder levels) and the xlsx to a destination folder; auto-detects the xlsx if not passed. Must be run with `uv run` from `scorecard/`. |
| `mark_reviewed.py` | Bulk-mark PAs as reviewed in the DB. |
| `correct_date.py` | Fix a mistyped game date (e.g. month/day swapped) without re-extracting: renames the game folder, `_cells.json`, `.html`, and scan image, patches `game.date` in the JSON and the DB `games` row (`date` + `raw_json_path`), regenerates the widget. The per-cell cache in `cells/` moves automatically with the folder rename. Date only — it does not touch opponent or home/away. |
| `manage_players.py` | CLI for fuzzy-matched player aliases (confirm, merge, list). |
| `_dump_cells.py` | Debug helper: prints cell cache as CSV (ri, ci, player, result, run, result_conf, run_conf, reread, adjusted, notes). |
| `test_rbi_attribution.py` | Unit tests for `_build_slot_data()` (PA construction + RBI attribution), `_check_rbi_leq_runs()`, and `classify_cell()`'s rbi_slot sentinel behavior (see #7). Run with `uv run python test_rbi_attribution.py` from `scorecard/`. No image needed; the two `classify_cell` tests monkeypatch `_call_api`/`_read_rbi_slot` instead of hitting the API. |
| `test_confidence.py` | Unit tests for `_score_cell()` and its wiring into `_build_slot_data()` (see #11 below). Run with `uv run python test_confidence.py`. No image or API needed. |

---

## Key design decisions

### Per-cell caching
Each cell result is persisted as `games/{stem}/cells/r{ri:02d}_c{ci:02d}.json` (1-based indices). `--reuse-cache` skips API calls for every cached cell. `api_error` cells always retry.

Cells removed by structural rules are stored as `removed:<rule> (<original_result>)`. On the next run they are **restored** to their original result so rules re-evaluate fresh — making rules stateless and idempotent.

### RBI slot detection
`_backfill_rbi_cells()` makes a focused VLM call on the bottom-left quadrant of every `run=True` cell missing an `rbi_slot` key. The quadrant contains either a single batting-order digit 1–9 (the batter who drove in the run) or a multi-character notation (SB/WP/PB/E# = no RBI). `thinking_budget=0` prevents Gemini thinking tokens from consuming the small `max_tokens` budget.

**RBI ↔ run coupling (#7).** `rbi_slot` is only ever meaningful on a `run=True` cell, so the coupling is enforced at both ends:
- *Read side* — `classify_cell()` only calls `_read_rbi_slot()` when `run` is true, and (this is the #7 fix) **omits the `"rbi_slot"` key entirely** when `run` is false, rather than writing `null`. That distinguishes "not applicable / never attempted" (key absent) from "attempted, no digit found" (key present, `null`) — the same value that `--reuse-cache` treats as "already read, skip." Before this fix a cell that was born `run=False` (so `rbi_slot` was written as `null`) and only later flipped to `run=True` by hole-reread, the HR constraint, or GT reconciliation would permanently look "already backfilled" and never get its digit read.
- *Ordering* — because of that, `_backfill_rbi_cells()` now runs **last**, after every pass that can set `run=True` (hole-reread, `_enforce_constraints`, `_apply_batting_rules`, `_enforce_gt_runs`, GT run reconciliation), right before `_build_slot_data()`. On `--reuse-cache`, `rbi_slot: null` is valid cached data; only cells with `run=True` and no `rbi_slot` key at all are backfilled.
- *Consumption side* — `_build_slot_data()` Phase 2 only reads `rbi_slot` off cells that already pass `cell.get("run")` (and have a non-null `result`), so a stray `rbi_slot` on a `run=False` cell — e.g. from hand-edited JSON — is inert.

**RBI attribution** happens in `_build_slot_data()` (pure function, unit-tested in `test_rbi_attribution.py`): a HR credits its batter 1 RBI at PA construction; every `run=True` cell with an `rbi_slot` digit credits the batter in that slot on their PA in the same inning. If `rbi_slot` equals the runner's **own** slot, nothing extra is credited: on a HR the scorer writes the batter's own number in the bottom-left, and that RBI is already counted (this self-reference produced 3 RBI on a 2-run HR — improvement #12; it only showed up when the VLM happened to read the digit, and vanished on a clean-slate run that read null). On a non-HR result a self-reference is impossible and is printed under "RBI attribution warnings" instead.

**RBI ≤ runs invariant (#7).** A run doesn't always imply an RBI (SB/WP/PB/E# advance the runner without one), but the reverse always holds: total RBI credited in an inning can never exceed runs scored that inning, since each run=True cell contributes at most one RBI credit (either the HR self-credit or one `rbi_slot` attribution, never both — see #12 above). `_check_rbi_leq_runs()` verifies this per inning after attribution and appends a warning (folded into the same "RBI attribution warnings" section) if it's ever violated — a sign of a bug, not a scorekeeping quirk, so it is not auto-corrected.

### Stolen base (SB) detection
`_backfill_sb_cells()` targets reached-base, non-out cells missing `sb_count`. Counts SB notations in the top-left/top-right/bottom-left quadrants (never bottom-right, which is RBI/out territory). `PlateAppearance.sb = int(cell.get("sb_count") or 0)`. Flows through `stats.py SB` → `export_season.py` (gold-highlighted season leader column, included in per-game and game-log sheets). CS was removed from the schema/exports — SB only.

### Voided / misscored cells
The VLM prompt instructs: if the bottom-right quadrant alone is crossed out with diagonal lines, or all four quadrants are crossed out, the PA was scored in the wrong cell — return `result: null` rather than guessing.

### Constraint enforcement
Applied before structural rules, in order:
1. **E (error) rule** — any result matching `^E\d*` is never an out.
2. **Out-run rule** — if a cell is an out (and not an error), `run` is forced to False.
3. **HR rule** — a `HR` result forces `run=True` (the batter always scores on his own home run); self-RBI is credited separately at PA construction (see RBI attribution below).

### Structural rules (batting rules)
1. **Isolation rule** — a PA with neither predecessor nor successor in the same inning is removed unless a neighbor is uncertain.
2. **3-out rule** — once 3 outs are recorded in batting order within an inning, all subsequent PAs in that inning are removed. Skipped for innings with uncertain cells.

`K-PB` (dropped third strike) is **not** an out for rules 1 and 3.

### Hole detection & the null-result/run=True consistency rule
A null cell sandwiched between two non-null cells in the same column is re-read (cyclic P9→P1 boundary suppressed — that's end-of-inning, not a hole). Separately, `_reread_hole_cells`-adjacent logic re-reads any `run=True, result=None` cell; **if the re-read still fails to produce a result, `run` is explicitly forced to `False`** in both the in-memory grid and the cache file. This matters because `PlateAppearance` construction (`extract_cells.py`, in the per-row PA-building loop) skips any cell with `result is None` entirely — such a cell contributes nothing to stats or the DB regardless of its `run` flag. Before this fix, a lingering `run=True/result=None` cell could make the per-inning integrity check report a phantom extra run that would never actually appear in the final output — always keep `run` and `result` consistent when hand-patching cells for this reason.

### Ground-truth enforcement
`_enforce_gt_runs()` runs after batting rules. For any inning where GT says R=0, all `run=True` cells are forced False. Over-counted innings are logged but not auto-corrected (needs a human to pick which cell is wrong).

### Confidence scoring (#11)
The VLM emits two integers per cell instead of a single "low"/"high" self-report: `result_conf` and `run_conf` (1-5, anchored rubric in `_CELL_SYSTEM` — 5 = every stroke unambiguous, 1 = barely legible). A survey of a full season's cache found the old binary field useless — 811/1413 cells had no confidence key at all, and every one of the remaining 602 said "high", including cells the pipeline itself had flagged as uncertain. `_score_cell()` (pure function, unit-tested in `test_confidence.py`) starts from that self-report — missing → 4, reason `"legacy"` — and docks it using evidence the pipeline already collects but used to discard: a successful hole/run reread (`cell["reread"]`, both -1), each structural rule correction (`cell["adjusted"]`, a deduplicated add-if-absent list written by `_enforce_constraints`, `_enforce_gt_runs`, `_apply_batting_rules`, the run-reconciliation `_recheck_run` call, and `classify_cell`'s `parse_retry`; each entry docks `run_conf` -1, except `parse_retry` which is itself an `adjusted` entry too), a `run=True` cell with no corroborating `rbi_slot` digit (`run_conf` -1), an inning whose R or H still mismatches GT after every pass (`inning_flags`, built from `_check_col()`'s final numbers and passed into `_build_slot_data()` — R mismatch docks `run_conf`, H mismatch docks `result_conf` on hit cells), and notes containing "ambig/unclear/guess/possibl/faint" (`result_conf` -1). Floors at 1, caps at 5. `PlateAppearance.confidence = min(result_conf, run_conf)`, plus the raw `result_conf`/`run_conf`/`conf_reasons` fields. `removed:`-tagged cells (isolation/3-out rules) and the cache-load restore path both now preserve these keys through a removal/restore cycle instead of dropping them — they describe VLM reads, not rule state.

`db.write_game` sets `needs_review = confidence <= config.yml`'s `review.threshold` (default 2); `conf_reasons` is stored `;`-joined in its own column. Old string confidence values ("high"/"low") are mapped onto the scale (5/2) both on JSON reimport (`models.PlateAppearance` validator) and in the DB (`init_db`'s idempotent migration). `review.py`'s default filter is unchanged (`needs_review=1`); `--max-conf N` widens it to any unreviewed PA with `confidence <= N` regardless of the stored threshold. `export_season.py`'s "Low Confidence" sheet queries `confidence <= review.threshold` directly and adds Confidence/Reasons columns.

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
| `plate_appearances` | `pa_id`, `player_id`, `game_id`, `inning`, `batting_order`, `result`, `run_scored`, `rbi`, `sb`, `bb`, `hp`, `sac`, `sf`, `confidence` (INTEGER 1-5), `conf_reasons` (`;`-joined), `needs_review` |

Duplicate detection: on reimport, any existing game with the same date + opponent is deleted before re-inserting. Fuzzy player matching uses `fuzzy.auto_match_threshold` in `config.yml`. Review threshold (`needs_review` cutoff) uses `review.threshold` in `config.yml` — see [Confidence scoring (#11)](#confidence-scoring-11).

---

## Re-running a game from cache

Pass the **same `--innings N`** as the original run (visible in `{stem}_run.log`, e.g. Urbanus was `--innings 6`). A different value makes the column count differ from `cells/_layout.json`, which is then re-detected and **overwritten**, and stray `r##_c##.json` files are created for the extra columns. Player stats are unaffected (extra columns are empty) but the folder is left inconsistent.

## Known-stale things to watch for

- The project's saved auto-memory (outside this repo) still references an older `process_game.py` / `extract_rows.py` / Anthropic-vision architecture from an earlier iteration of this project — that pipeline no longer exists. Trust this file and the code over old memory notes if they conflict.
- `players.txt` should have one player per line, no duplicates, trailing newline present — check it after any bulk name-correction session.
