# Backend pipeline — architecture reference

This document describes the data flow and key design decisions in the scorecard digitization pipeline. It is a developer reference; for the user-facing run instructions see [README.md](README.md).

---

## High-level flow

```
Scan image (JPG/PNG)
        │
        ▼
1. Grid detection        probe_grid.py / detect_grid()
        │                → row + column coordinates
        ▼
2. Per-cell VLM          extract_cells.py / classify_cell()
        │                → 81 raw cell dicts (result, run, notes, confidence)
        │                   cached in images/_cache/cells/{stem}/
        ▼
3. Structural rules      _apply_batting_rules()
        │                → out-run, isolation, 3-out enforcement
        ▼
4. GT enforcement        _enforce_gt_runs()
        │                → fix impossible runs against ground-truth totals
        ▼
5. Reconciliation        main loop
        │                → focused re-check when extracted R < GT R
        ▼
6. Integrity checks      _check_row(), _check_pa_sequence(), _check_col()
        │                → per-player and per-inning cross-checks printed
        ▼
7. JSON export           GameExtraction (models.py)
        │                → images/data/raw/{stem}_cells.json
        ▼
8. DB write              db.py / write_game()        ← NOT YET WIRED
        │                → scorecard/data/season.db
        ▼
9. Excel export          export_season.py            ← runs from DB
           → stats.xlsx
```

---

## Files

| File | Role |
|---|---|
| `extract_cells.py` | Main entry point for the new per-cell pipeline. Grid detection → VLM → rules → checks → JSON. |
| `probe_grid.py` | OpenCV grid detector: finds row separators and inning column boundaries from the scan. |
| `probe_lines.py` | Older line-detection helper (used by `split_rows.py`). |
| `models.py` | Pydantic models: `GameExtraction`, `LineupSlot`, `PlateAppearance`, `InningTotals`, etc. |
| `db.py` | SQLite layer: `init_db`, `write_game`, `find_duplicate_game`, fuzzy player matching. |
| `extract.py` | **Old pipeline**: full-scorecard VLM prompt (Anthropic). Used by `process_game.py`. |
| `extract_rows.py` | **Old pipeline**: per-row VLM extraction with ESRGAN enhancement. Superseded by `extract_cells.py`. |
| `process_game.py` | **Old pipeline** orchestrator: calls `extract.py` → `db.py`. Still functional. |
| `export_season.py` | Reads DB and writes multi-sheet Excel workbook (`stats.xlsx`). |
| `review.py` | CLI to review low-confidence PAs in the DB. |
| `mark_reviewed.py` | Marks PAs as reviewed in the DB; re-exports workbook. |
| `manage_players.py` | CLI for fuzzy-matched player aliases (confirm, merge, list). |
| `enhance.py` | ESRGAN 4× upscaling for the old row-based pipeline. |
| `split_rows.py` | Splits a full scorecard into per-player row images (old pipeline). |
| `stats.py` | Computes derived stats (AVG, OBP, SLG, wOBA, OPS+, etc.) from DB rows. |
| `_dump_cells.py` | Debug helper: prints the cell cache as CSV (ri, ci, player, result, run, confidence, notes). |

---

## Key design decisions

### Per-cell caching
Each cell result is persisted as `images/_cache/cells/{stem}/r{ri:02d}_c{ci:02d}.json` (1-based indices). Running with `--reuse-cache` skips API calls for every cached cell. `api_error` cells always retry; `parse_error` cells attempt salvage from partial JSON.

Cells previously removed by structural rules are stored as `removed:<rule> (<original_result>)`. On the next run they are **restored** to their original result so rules re-evaluate fresh. This makes rules stateless and idempotent.

### Structural rules (batting rules)
Applied after all cells are classified, in order:

1. **Out-run rule** — K, F#, groundouts, DP, SAC, SH, SF cannot have `run=True`. Immediate cache write.
2. **Isolation rule** — A PA with neither predecessor nor successor in the same inning is physically impossible (batting order is continuous). Removed unless either neighbor is uncertain (api_error).
3. **3-out rule** — Once 3 outs are recorded in batting order within an inning, all subsequent PAs in that inning are removed. Skipped entirely for innings with any uncertain cells.

`K-PB` (dropped third strike) is **not** an out for purposes of rules 1 and 3. See the VLM prompt for the K-PB detection logic.

### Ground-truth enforcement
`_enforce_gt_runs()` runs after batting rules. For any inning where GT says R=0, all `run=True` cells are forced False. Over-counted innings (extracted > GT) are logged but not auto-corrected (per-cell GT would be needed).

### PA cross-check identity
For a completed inning: `PA = 3 (outs) + R (runs) + LOB (left on base)`. Both 3 and LOB come from the ground-truth totals file. This is checked in `_check_col()`.

### PA sequencing invariant
Batting order is cyclic. Over a full game, player N cannot have more total PAs than player N−1. `_check_pa_sequence()` verifies this after all row checks.

### VLM model
Default: `gemini-2.5-flash` (set via `EXTRACTION_MODEL` env var or `--model`). The prompt (`_CELL_SYSTEM` in `extract_cells.py`) describes all valid PA result codes, K-PB detection, and supplementary notation conventions (WP/PB/SB in sub-cells = baserunner advancement only, not PA result).

---

## Ground-truth files

Placed in `images/ground_truth/` next to the scans:

| File | Content |
|---|---|
| `{stem}_totals.txt` | Per-inning: R, H, E, LOB. One row per inning. Hand-edited after scanning the bottom strip. |
| `{stem}_stats.txt` | Per-player: batting slot, H, AB. One row per player; substitutions get two rows with the same slot number. |

These files are used for enforcement and cross-checks only — never for individual cell classification.

---

## DB integration status

`extract_cells.py` outputs a `GameExtraction` JSON (same schema as the old pipeline) to `images/data/raw/{stem}_cells.json`. The `db.py` `write_game()` function accepts a `GameExtraction` object, but **the call is not yet wired** from `extract_cells.py`. The old `process_game.py` → `extract.py` → `db.py` path still works independently.

To complete the bridge: call `write_game(game, conn)` from the end of `extract_cells.py main()` unless `--dry-run` is passed.

---

## Cache directory layout

```
images/
  scans/              ← input scan images
  ground_truth/       ← hand-edited GT files
  _cache/
    cells/
      {stem}/
        r01_c01.json  ← cell (row 1, col 1), 1-based
        r01_c02.json
        ...
    step1/            ← legacy row-level VLM output (old pipeline)
  data/
    raw/              ← per-game GameExtraction JSON output
scorecard/
  data/
    season.db         ← SQLite (populated by old pipeline only, for now)
```
