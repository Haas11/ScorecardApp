# CLAUDE.md — pipeline architecture & agent notes

Developer/agent reference for the scorecard digitization pipeline. For run instructions see [README.md](README.md).

**Security context (org policy):** subject to CMMC, CRA, PCI, TISAX, EU AI regulations. `scorecard/.env` holds the Gemini API key and is gitignored — never commit API keys or read `.env` contents into chat/output.

---

## High-level flow

```
Scan image (JPG/PNG)
        │
        ▼
1. Grid detection        rectify.py / detect_grid_lattice()  (legacy: probe_grid.py / detect_grid())
        │                → straighten photo if skewed, lattice-fit row + column boundaries
        │                   (legacy detector on manual override / --legacy-grid / low fit score)
        ▼
2. Name detection        extract_cells.py / _detect_row_names()
        │                → VLM reads left info strip per row, fuzzy-matched to roster
        │                → substitutions: second name triggers sub_inning prompt
        │                   cached in games/{stem}/cells/_names.json (ALWAYS reused if present,
        │                   independent of --reuse-cache — delete or use --reset-names to redo)
        ▼
3. Per-cell VLM          extract_cells.py / classify_cell()
        │                → ink-free cells skipped (no API); out-circle cross-check and
        │                   bottom-left-ink run trigger inside classify_cell (see Reliability checks)
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
9. Reconciliation        _reconcile_gt_runs()  (#1)
        │                → extracted R < GT R: structurally forced runs (runner ahead of a
        │                   scorer, all 3 outs are PA outs), then ranked VLM re-check
        │                → extracted R > GT R: remove lowest-run_conf scorer only on a
        │                   clear confidence gap; HR never removed; else flag for review
        ▼
10. Integrity checks     _check_row(), _check_pa_sequence(), _check_col()
        │                → per-player and per-inning cross-checks printed to terminal
        │                → run counts here EXCLUDE cells with result=None even if run=True,
        │                   matching what actually reaches PlateAppearance/stats
        ▼
10b. Second read         _verify_candidates() / _verify_uncertain_cells()
        │                → uncertain cells re-read once; disagreement docks confidence
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
| `app.py` | Streamlit GUI (#22), `uv run streamlit run app.py`, localhost only (`.streamlit/config.toml`, telemetry off). Holds no pipeline logic: data changes go through `scorecard.py` subprocesses. Review page maps each PA to its cache cell (k-th non-empty cell of row×inning = k-th PA) and re-detects the grid (cached) for the crop. **Save** writes the PA with `conf_reasons=["hand-checked"]`, then `sync-edits` sets `hand_checked: true` on the cache cell (dropping any `verify`), and `_score_cell` returns 5/5 for such cells. |
| `scorecard.py` | Single entry point (#21): plain-language subcommands mapped lazily onto the scripts below (`read-scorecard`→extract_cells, `reimport-game`→reimport, `sync-edits`→reimport --sync-cells, `reread-season`→crawl, `review-cells`, `fix-date`, `players`, `export-excel`, `publish`, `draw-widget`, `test-accuracy`→eval_cells, `run-tests`). Add new user-facing commands to its `_COMMANDS` table. |
| `gamepaths.py` | `resolve_game()` (#20): every per-game command accepts the game folder, any file inside it, the scan, or just the game name. Use it for any new per-game command. |
| `extract_cells.py` | Main pipeline entry point. Grid → names → VLM → backfills → constraints → rules → checks → JSON → DB → HTML. |
| `rectify.py` | Default grid detector (#16): perspective/skew rectification + lattice fit on a printed-line mask. Works on phone photos. See [Grid detection](#grid-detection-rectifypy-default-probe_gridpy-legacy). |
| `probe_grid.py` | Legacy OpenCV grid detector: finds row separators and inning column boundaries; supports manual overrides. Used only as fallback. |
| `models.py` | Pydantic models: `GameExtraction`, `LineupSlot`, `PlateAppearance`, `InningTotals`, etc. |
| `db.py` | SQLite layer: `init_db`, `write_game`, `find_duplicate_game`, fuzzy player matching. |
| `stats.py` | Derived stat calculations: AVG, OBP, SLG, OPS, BABIP, ISO, wOBA, RC, OPS+, BB/K, AB/HR. SB counted; CS removed from schema. |
| `export_season.py` | Reads DB, writes multi-sheet Excel workbook with color-scaled conditional formatting. No CS column. |
| `reimport.py` | Reimport one `_cells.json` or all games (`--all`) into DB + regenerate HTML. `--sync-cells` reverse-writes a hand-edited `_cells.json` back into the per-cell cache (preserves `rbi_slot`, `reread`, `adjusted`; writes back `result_conf`/`run_conf` if present) so `--reuse-cache` doesn't clobber manual fixes. |
| `crawl.py` | Loops game folders under a games dir, re-running `extract_cells.py --reuse-cache --yes` per game (finds the scan in the sibling `scans/` folder). Use to backfill a newly-added field across already-analyzed games. |
| `review.py` | Interactive CLI to review and correct low-confidence PAs (default: `needs_review=1`; `--max-conf N` widens to any unreviewed PA with confidence <= N; `--all` shows every PA). GAME is a positional argument (full name, path, or name fragment, via `db.find_game_ids`). A corrected or confirmed PA is written as hand-checked (confidence 5, `conf_reasons=["hand-checked"]`) to the DB and `_cells.json`, the latter matched on the nth PA of slot × inning (wrapped innings have two). After the session it runs `sync_cells_from_json` so the cache gets `hand_checked`. |
| `render_widget.py` | Standalone HTML widget renderer (color-coded scorecard grid + per-player stats + SB indicator). |
| `publish.py` | Copies `*.html` (root + up to 2 subfolder levels) and the xlsx to a destination folder; auto-detects the xlsx if not passed. Must be run with `uv run` from `scorecard/`. |
| `mark_reviewed.py` | Bulk-mark PAs as reviewed in the DB. |
| `correct_date.py` | Fix a mistyped game date (e.g. month/day swapped) without re-extracting: renames the game folder, `_cells.json`, `.html`, and scan image, patches `game.date` in the JSON and the DB `games` row (`date` + `raw_json_path`), regenerates the widget. The per-cell cache in `cells/` moves automatically with the folder rename. Date only — it does not touch opponent or home/away. |
| `manage_players.py` | CLI for fuzzy-matched player aliases (confirm, merge, list). |
| `_dump_cells.py` | Debug helper: prints cell cache as CSV (ri, ci, player, result, run, result_conf, run_conf, reread, adjusted, notes). |
| `test_rbi_attribution.py` | Unit tests for `_build_slot_data()` (PA construction + RBI attribution), `_check_rbi_leq_runs()`, and `classify_cell()`'s rbi_slot sentinel behavior (see #7). Run with `uv run python test_rbi_attribution.py` from `scorecard/`. No image needed; the two `classify_cell` tests monkeypatch `_call_api`/`_read_rbi_slot` instead of hitting the API. |
| `test_confidence.py` | Unit tests for `_score_cell()` and its wiring into `_build_slot_data()` (see #11 below). Run with `uv run python test_confidence.py`. No image or API needed. |
| `eval_cells.py` | VLM eval set (#0): re-reads every cell of the hand-verified games (Grizzlies 2026-07-12, Urbanus 2026-08-23) with the current prompts and reports result/run/RBI/presence accuracy against their cache labels. `--check` = no API (label stats and a cache vs `_cells.json` consistency check). Predictions go to `eval_out/` (gitignored). Gemini varies by 2–3 cells between runs, so compare more than one run. Run it before and after any prompt/crop change. |
| `test_grid.py` | Grid-geometry regression (#16): runs `detect_grid()` on the hand-verified baseline games (Grizzlies 2026-07-12, Urbanus 2026-08-23) and compares row/col boundaries to `grid_golden.json` (tolerance 4% of a cell). `--update` rewrites the golden — only do that after visually checking the debug images. No API. Run with `uv run python test_grid.py`. |
| `test_gamepaths.py` | Unit tests for `gamepaths.resolve_game()` (#20) on a temp folder layout. **Run all offline tests at once with `uv run python scorecard.py run-tests`** (also runs `eval_cells.py --check`). |
| `test_gt_runs.py` | Unit tests for `_reconcile_gt_runs()` (#1): forced runs, ranked re-check order, over-count gating, cyclic batting order. Uses a fake re-check callable; no image or API needed. Run with `uv run python test_gt_runs.py`. |

---

## Key design decisions

### Per-cell caching
Each cell result is persisted as `games/{stem}/cells/r{ri:02d}_c{ci:02d}.json` (1-based indices). `--reuse-cache` skips API calls for every cached cell. `api_error` cells always retry.

**The cache is tracked in git, on purpose.** The `.gitignore` pattern for it names an old folder (`Quick 2026 data/`) and has never matched, and that turned out to be the right outcome: the user hand-corrects `r##_c##.json` files against the paper scorecard, and the cache is what every `--reuse-cache` run and `crawl.py` rebuild from. Consequences: (1) after any crawl, `git diff -- "Quick 2026/games"` shows every cache and `_cells.json` change a run made, and `git checkout` reverts a game; (2) commit the game folders after a manual correction session so the corrections are versioned — and **sync hand edits made in `_cells.json` back into the cache** with `reimport.py --sync-cells`, otherwise the next `--reuse-cache` run or crawl regenerates the JSON from the cache and silently drops them (this is exactly what happened to Domstad Dodgers 2026-03-29 in the 2026-09-03 crawl: 22 hand-fixed PAs in the JSON, 15 in the cache-derived output — the JSON was restored from git); (3) passes that can overwrite a hand-edited cell without a VLM re-read of its result are the hole re-read (a manually nulled cell sandwiched between two PAs is re-read), the run-no-result re-read, and the GT run reconciliation (#1) — check the diff for those after a crawl.

Cells removed by structural rules are stored as `removed:<rule> (<original_result>[, run])` (`_removed_note` / `_parse_removed_note`). On the next run they are **restored** to their original result *and run flag* so rules re-evaluate fresh — making rules stateless and idempotent. The run flag is recorded since the 2026-09-03 crawl; before that a restored HR came back `run=False`, the HR constraint fired on it and wrote an `adjusted` flag onto a cell that was then removed again, and a removed-then-restored scorer lost its run. Old-format notes are still accepted (run is assumed only for a HR).

Cells in buffer columns whose inning maps beyond `--innings` (a 9-inning game has no inning 11 — those are reads of the card's right-hand totals area) are dropped by `_build_slot_data(max_inning=innings)` with a warning under "RBI attribution warnings"; they never become plate appearances. Found on Almere 2026-06-07, which carried a phantom 1B in "inning 11".

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

### Reliability checks added 2026-09-28 (all measured with `eval_cells.py`)
- **Gemini thinking budget.** Gemini 2.5 spends thinking tokens out of `max_output_tokens`. With the old `max_tokens=400` and dynamic thinking, nearly every cell answer was cut off mid-JSON (`finish_reason=MAX_TOKENS`). The salvage parser recovered result/run but lost `result_conf`/`run_conf`: only 28/135 reads had them, so #11 scoring was mostly "legacy". `classify_cell` now uses `_CELL_THINKING_BUDGET=1024`, `_CELL_MAX_TOKENS=1424`. Evaluation: results 79→81/84 and runs 84/84 (with the checks below), conf on 135/135, about 3× slower per call. Every other short Gemini call (`_recheck_run`, totals, names, sub-inning) passes `thinking_budget=0`. **`_recheck_run` (max_tokens=10) could previously only ever return False**, so the GT run re-check (#1) never fired before this fix. **Always pass `thinking_budget` on new Gemini calls.**
- **Empty-cell skip.** Before classification, cells whose inner 84% has < `_EMPTY_INK_MAX` (0.5%) coloured-pen pixels (`_pen_ink`) are cached as `{"result": null, "notes": "no_ink"}` with no API call. Evaluation set: 38/135 skipped, 0 real PAs lost (every PA has ≥ 3.1% ink); `eval_cells.py --check` re-verifies this offline. Guard: skipping is disabled for the game when fewer than 3 × innings cells are clearly inked (≥ 3%), e.g. a pencil or black-pen card with no coloured ink. The hole re-read is still the safety net.
- **Out circle cross-check.** `_circle_state()` (HoughCircles on the pen-ink mask + inked share of the circumference) returns `circle` / `none` / `unsure`. Only circles that **enclose the cell centre** by ≥ `_CIRCLE_ENCLOSE` (0.06 × cell) count: a runner-out circle sits in one quadrant. Without this rule, Kinheim 2026-09-04 P2 inn 1 (a circled "64" top-right plus a 1B) was flipped from a correct 1B to 6-4. Cut-offs 0.74/0.55 gave 0 contradictions on the evaluation set, deciding 23/32 outs and 51/52 non-outs. When it contradicts the read's out/not-out, `classify_cell` re-reads once with the finding as a hint. If the re-read agrees it is taken (`circle:<old>-><new>`, result_conf -1); otherwise the first read is kept with `circle_mismatch` (result_conf -2 → review). New reads only.
- **Second read of uncertain cells (suggestion 5).** After run reconciliation, `_verify_candidates()` picks PA cells whose self-reported `result_conf` or `run_conf` is ≤ `_VERIFY_CONF_MAX` (4, because the VLM almost never reports lower), cells with `circle_mismatch`, and at-bat cells in an inning whose H still mismatches GT. Legacy cells with no self-report are only picked by the last two. It sorts them lowest confidence first, caps at 30 per game, and `_verify_uncertain_cells()` re-reads each with a plain `classify_cell(..., circle_check=False)` (no circle hint, so a hint-induced flip in the first read shows up as a disagreement instead of being repeated), using `--verify-model` (default: `--model`). Only `cell["verify"] = {model, result, run}` is stored, in the grid and the cache, and the first read is never replaced. `_verify_agreement()` compares at scoring time against the cell's current values, so a hand correction or a later pass never leaves a stale verdict. It compares on `_stat_class`: fielder digits are ignored, hit type/K/fly/ground/E are not. Run is not compared when a later pass changed it (`run->` in `adjusted`). In `_score_cell`, a disagreement gives result_conf -2 (`verify:result <first>|<second>`) or run_conf -2 (`verify:run`), which usually lands in review; agreement gives result_conf +1 (`verify:agree`). Cells already verified with the same model are skipped, so re-runs cost nothing. `--no-verify` disables the pass. Evaluation over 3 runs: 51 candidates (about 8 per game); the second read disagreed on 5, catching 4/4 real errors among them, with 1 false alarm. `gemini-2.5-pro` as verifier returned nothing usable with the current call settings (failed reads are skipped), so it is untested.
- **Per-player GT (card columns 11–12): opt-in (`--read-player-stats`), off by default.** The columns are often empty and include sac flies, which the pipeline doesn't track yet. An existing `_stats.txt` is always loaded. With the flag, if no `{stem}_stats.txt` exists, `_detect_player_stats()` reads each slot's H-AB and AVG lines (one call per slot, `thinking_budget=0`). It keeps a slot only when every line's H/AB matches the written AVG (`_stats_line_ok`), sums starter + subs, and writes `_stats.txt` (`slot H AB`; edit to correct). It read 18/18 slots correctly on the two verified games. `_check_row` now returns the mismatching stats, and for subbed slots also prints a `= slot total` line checked against the GT. `row_flags` goes into `_build_slot_data(row_flags=...)` → `_score_cell(..., slot_flags)`: at-bat cells in a mismatching slot get result_conf -1 (`slot_H_mismatch`/`slot_AB_mismatch`), and -1 more where a slot H mismatch crosses an inning H mismatch (`row_and_inning_H_mismatch`), which is the likely misread cell.

### Bottom-left ink run trigger (#6)
The VLM misses clear run dots. The bottom-left (home-plate) quadrant records how a runner reached home, so pen ink there means a run. In `classify_cell()`, a non-out cell the VLM read as `run=False` with `_bottom_left_ink()` > `_BL_INK_TRIGGER` (2% saturated pixels; printed grid lines are grey) gets one focused `_read_bottom_left_mark()` call. A `digit` or `notation` answer sets `run=True`, `rbi_slot` = the digit, and adds the `bl_ink:run->True` adjusted flag. A `stray` answer (slash or connector line from another cell) or `blank` leaves the cell unchanged. This applies only to new reads.

### Inning wrap detection (#23)
`_resolve_col_to_inning()` (unit-tested in `test_wrap.py`). Primary signal: the totals (GT) row. A totals cell with no handwriting (`_totals_blank`, a local threshold minus printed lines, < 2%; written cells measure ≥ 6.9%, phone photo included), under a column with ≥ n_rows−2 PAs and with PAs in the next column, is a wrap. A *written* totals cell rules a wrap out. This is used only when the scorekeeper filled in at least half the played columns' totals (Grizzlies 2026-05-10 left the row almost empty). Otherwise it falls back to the grid heuristic (`_detect_wrap_from_grid`: all batted, < 3 outs), which a single misread out breaks. On Herons 2026-09-27 that heuristic missed the inning-4 wrap and invented one at col 8. `_layout.json`'s `col_to_inning` still wins when present (it may be hand-edited), but a mismatch with the fresh detection is printed as a warning. To re-detect, remove that key and the auto-extracted `_totals.txt`.

### Reached-base check for "no PA" reads
Everything written in a cell is about that batter: a notation in the 2nd/3rd/home quadrants (SB, WP, PB, E#, a digit, a runner-out circle) means he reached base, so a null result is wrong. In `classify_cell()`, a null read whose top-left/top-right/bottom-left quadrant has > `_BASE_INK_TRIGGER` (10%) pen ink gets one focused `_read_reached_base()` call. That call first decides whether there are written characters (stray lines don't count), then reads a non-out result from the bottom-right. Success sets the result, caps `result_conf` at 3 and adds `reached:null-><result>`. The eval set's 51 empty cells peak at 8.1%; Herons 2026-09-27 r08_c09 (circled 85, E24, a sloppy BB read as "P6") is at 17.8%. Putting this rule in `_CELL_SYSTEM` instead made two stray-mark cells phantom PAs, and in two runs reads of other cells got worse (73-74/84). So it lives in the focused call. New reads only.

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

### Ground-truth enforcement and run reconciliation (#1)
`_enforce_gt_runs()` runs after batting rules. For any inning where GT says R=0, all `run=True` cells are forced False. Every other mismatch is handled by `_reconcile_gt_runs()` (pure apart from grid/cache writes, unit-tested in `test_gt_runs.py` with a fake VLM re-check), which runs after the integrity checks and before the RBI backfill. It works on the inning's **batting sequence** (rows rotated to the lead-off batter from `last_batter_by_inning`, overflow columns appended) because baseball structure gives priors a per-cell read cannot:

- **Under-count, tier 1 (forced).** Runners cannot pass each other, so a batter who reached base *ahead of the last scorer* must have scored or been retired on the bases. When the inning's 3 outs are all plate-appearance outs (no `DP`, no "runner out" note, lead-off batter known), nobody was retired on the bases, so those runners scored — assigned with no VLM call, tagged `gt:run->True(forced)`. If more runners are forced than runs are missing, a `run=True` cell is probably wrong instead: nothing is assigned, the inning is flagged.
- **Under-count, tier 2 (ranked VLM re-check).** Remaining non-out, non-scoring cells are re-read with `_recheck_run` in order of lowest `run_conf` first (least sure of "no run"), then earliest in sequence. Only a positive re-read assigns (`recheck:run->True`).
- **Under-count, tier 3.** Still short → candidates listed with their `run_conf`, nothing invented; the inning keeps its `R` flag so every cell in it is docked for review.
- **Over-count.** Candidate scorers ranked by `run_conf` ascending; a `HR` is never a candidate; a scorer sitting behind a non-scoring runner in an airtight inning is ranked one notch lower. The lowest `m` are removed (`gt:run->False`) only when there is a gap of at least `_GT_RUN_REMOVE_GAP` (2) between the last removed and the first kept, or when every non-HR scorer has to go. Otherwise flagged. Legacy caches (no self-report, everything scores 4) therefore never auto-remove — by design.

All flips go through `_add_adjusted`, so `_score_cell` docks them and `--reuse-cache` re-application is idempotent.

### Confidence scoring (#11)
The VLM emits two integers per cell instead of a single "low"/"high" self-report: `result_conf` and `run_conf` (1-5, anchored rubric in `_CELL_SYSTEM` — 5 = every stroke unambiguous, 1 = barely legible). A survey of a full season's cache found the old binary field useless — 811/1413 cells had no confidence key at all, and every one of the remaining 602 said "high", including cells the pipeline itself had flagged as uncertain. `_score_cell()` (pure function, unit-tested in `test_confidence.py`) starts from that self-report — missing → 4, reason `"legacy"` — and docks it using evidence the pipeline already collects but used to discard: a successful hole/run reread (`cell["reread"]`, both -1), each structural rule correction (`cell["adjusted"]`, a deduplicated add-if-absent list written by `_enforce_constraints`, `_enforce_gt_runs`, `_apply_batting_rules`, the run-reconciliation `_recheck_run` call, and `classify_cell`'s `parse_retry`; each entry docks `run_conf` -1, except `parse_retry` which is itself an `adjusted` entry too), a `run=True` cell with no corroborating `rbi_slot` digit (`run_conf` -1), an inning whose R or H still mismatches GT after every pass (`inning_flags`, built from `_check_col()`'s final numbers and passed into `_build_slot_data()` — R mismatch docks `run_conf`, H mismatch docks `result_conf` on hit cells), and notes containing "ambig/unclear/guess/possibl/faint" (`result_conf` -1). Floors at 1, caps at 5. `PlateAppearance.confidence = min(result_conf, run_conf)`, plus the raw `result_conf`/`run_conf`/`conf_reasons` fields. `removed:`-tagged cells (isolation/3-out rules) and the cache-load restore path both now preserve these keys through a removal/restore cycle instead of dropping them — they describe VLM reads, not rule state.

`db.write_game` sets `needs_review = confidence <= config.yml`'s `review.threshold` (default 2); `conf_reasons` is stored `;`-joined in its own column. Old string confidence values ("high"/"low") are mapped onto the scale (5/2) both on JSON reimport (`models.PlateAppearance` validator) and in the DB (`init_db`'s idempotent migration). `review.py`'s default filter is unchanged (`needs_review=1`); `--max-conf N` widens it to any unreviewed PA with `confidence <= N` regardless of the stored threshold. `export_season.py`'s "Low Confidence" sheet queries `confidence <= review.threshold` directly and adds Confidence/Reasons columns.

### PA cross-check identity
For a completed inning: `PA = 3 (outs) + R (runs) + LOB`. Checked in `_check_col()` using ground-truth totals. `_check_col` and the run-reconciliation loop both only count a cell as a "run" if it has `run=True` **and** a non-null `result` — matching the final PlateAppearance filter (see above).

### Player name and substitution detection
The left info strip of each row is cropped and sent to the VLM. The active player for a given cell is resolved from `_names.json` using `sub_innings`: the first name is the starter; subsequent names apply from the listed inning onwards. A detected sub-slot name that's too short after stripping dots/spaces (< 4 significant chars) and doesn't fuzzy-match the roster is treated as VLM noise and silently dropped rather than prompted — real abbreviated names are at least "A. X" (4 chars).

`players.txt` is appended to when a new player is chosen interactively; the append now guards against a missing trailing newline on the existing file (a prior bug concatenated a new name onto the last line, then repeated appends across sessions produced a badly duplicated roster — always eyeball `players.txt` after a bulk name-correction session).

### Grid detection (rectify.py default, probe_grid.py legacy)
**Default: `rectify.detect_grid_lattice()` (#16).** (1) Adaptive threshold → long-line mask (short morphological kernels, so a few degrees of tilt survive) → `polyfit` per line component → quad from the outermost long H/V lines → homography. The image is warped only when the corner shift exceeds 5% of a cell. Clean scans need < 6px, so they are left untouched; the warped image is saved as `games/{stem}/{stem}_rectified.jpg` and **all crops come from it**. (2) `fit_grid()` fits a lattice to the line mask (handwriting and lighting are gone there). Rows: the name strip is split in thirds and the inning area in halves, so row boundaries are exactly the lines present in both. A brute-force `(y0, pitch)` search over their product finds the 11 boundaries, and each is snapped to the raw inning-area profile. Columns: the `#` and row-number columns left of inning 1 are each about half a cell wide and mimic the half-pitch pattern, so column 1 is found by a discrete rule. It is the leftmost boundary (a line in both the header band and the body) whose next half-step is a body-only sub-column line; from there it steps by 2 × the half-pitch and snaps. A fit score below 0.8 falls back to the legacy detector. `test_grid.py` guards it: the two hand-verified games must match their legacy geometry within 4% of a cell, and the two phone photos (Vennep 06-28, Kinheim 09-04) their visually checked fits.

**Legacy: `probe_grid.detect_grid()`**, used with `--legacy-grid`, with any of `--grid-start`/`--grid-width`/`--cell-height` (given or persisted in `run_args`), or on a rejected lattice fit.
V-line and H-line detection use OpenCV projection + gap analysis, with a bimodal check (full-row vs sub-row divider gaps in ~2:1 ratio). The bimodal cell-size is computed as `lower_med * 2` (sub-row divider height × 2), **not** `upper_med` — the upper cluster can undershoot when some full-row gaps are noisy or partially missed. Outlier full-row gaps (>3× median, from missed H-lines) fall back to the median instead of dominating the estimate.

For difficult scans (low-res, landscape, skewed, wide player-info columns), manual overrides bypass detection:
- `left_skip_frac` / `--left-skip` — fraction of image width to skip before V-line detection (increase for wide player-info areas).
- `grid_start` + `grid_col_width` / `--grid-start` + `--grid-width` — force a uniform column grid, skipping V-line detection entirely.
- `cell_height` / `--cell-height` — override the detected row height directly when bimodal detection still gets it wrong.
- Left column extrapolation: if fewer inning columns are detected than expected, missing columns are synthesized by stepping backwards from the first detected column (not forwards from an assumed start — that produced gap mismatches).

Example real override that was needed for a 1345×948px landscape/low-res scan (2026-28-06 Vennep Flyers): `--innings 10 --grid-start 455 --grid-width 75 --left-skip 0.35`. These overrides — and `--innings`/`--n-player-rows`/`--cell-height` — only need to be given once; see [Re-running a game from cache (#17)](#re-running-a-game-from-cache-17).

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

## Re-running a game from cache (#17)

`--innings`, `--n-player-rows`, `--left-skip`, `--grid-start`, `--grid-width`, and `--cell-height` all default to `None` on the CLI now, not a hardcoded value. Whichever of them isn't given explicitly this run falls back to the value stored in `cells/_layout.json`'s `run_args` key from a *prior* run of this exact game, and only then to the hardcoded default (`innings=9`, `n_player_rows=10`, `left_skip_frac=0.05`, the rest `None`). Resolution happens before grid detection, so a bare `--reuse-cache --yes` (what `crawl.py` passes) reuses whatever this game actually needed — it no longer silently re-detects at the default and overwrites `_layout.json` + leaves stray `r##_c##.json` files for a game that needed `--innings 6` or a manual grid override. `run_args` is rewritten every run (not just when `col_to_inning` is freshly detected), so passing an explicit override this run updates what the *next* bare `--reuse-cache` run reuses. `--active-players` is deliberately **not** persisted this way — it legitimately varies per game (short lineup, etc.), unlike grid geometry.

Before this fix, a bare `--reuse-cache` re-run without the original flags would make the column count differ from `_layout.json`, triggering a silent re-detect-and-overwrite plus stray cache files for the extra columns (safe for stats — the extra columns are empty — but left the folder inconsistent). That's what blocked `crawl.py` from running unattended over the whole season: 4 of 15 games (Vennep Flyers 2026-06-28: 10 innings + manual grid override; Herons 2026-07-03 and Grizzlies 2026-07-12: 7 innings; Urbanus 2026-08-23: 6 innings) needed a non-default flag. Their `_layout.json` files were backfilled with the correct `run_args` from their `_run_log`/this file's documented overrides so the fix protects them starting now, not only after their next manual re-run.

## Known-stale things to watch for

- The project's saved auto-memory (outside this repo) still references an older `process_game.py` / `extract_rows.py` / Anthropic-vision architecture from an earlier iteration of this project — that pipeline no longer exists. Trust this file and the code over old memory notes if they conflict.
- `players.txt` should have one player per line, no duplicates, trailing newline present — check it after any bulk name-correction session.
