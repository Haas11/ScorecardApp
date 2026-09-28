"""
Grid-geometry regression test (#16).

Runs the pipeline's grid detection (lattice fit, or the legacy detector when the
game's _layout.json has manual overrides) and compares row/column boundaries
against grid_golden.json.

BASELINE_GAMES: hand-verified cell data; golden = the geometry the verified reads
were cropped from (snapshotted from the legacy detector). The lattice detector
must reproduce it within tolerance.
PHOTO_GAMES: phone photos; golden = lattice fit in rectified-image coordinates,
checked by eye on the debug image (2026-09-28), not hand-verified cell data.

No API calls. Run from scorecard/:
    uv run python test_grid.py            # check
    uv run python test_grid.py --update   # rewrite grid_golden.json from current detector
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

from probe_grid import detect_grid
from rectify import detect_grid_lattice

HERE = Path(__file__).parent
GOLDEN = HERE / "grid_golden.json"
SEASON = HERE.parent / "Quick 2026"
BASELINE_GAMES = [
    "2026-07-12 Grizzlies (Home)",
    "2026-08-23 Urbanus (Home)",
]
PHOTO_GAMES = [
    "2026-06-28 - Vennep Flyers (Home)",
    "2026-09-04 Kinheim (Away)",
]
# Max boundary deviation, as a fraction of cell size (~4-5px on a 2700px scan).
TOL_FRAC = 0.04


def _run(stem: str, legacy: bool = False) -> dict:
    layout = SEASON / "games" / stem / "cells" / "_layout.json"
    a = json.loads(layout.read_text()).get("run_args", {}) if layout.exists() else {}
    innings = a.get("innings") or 9
    rows = a.get("n_player_rows") or 10
    scan = str(SEASON / "scans" / f"{stem}.jpg")
    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as tmp:
        res = None
        if not legacy:
            res = detect_grid_lattice(scan, n_player_rows=rows, rectified_out=str(Path(tmp) / "r.jpg"))
        if res is None:
            res = detect_grid(scan, n_player_rows=rows, n_inning_cols=innings,
                              left_skip_frac=a.get("left_skip_frac") or 0.05)
    row_tops, row_bottoms, _, _, col_lefts, cell_size = res[:6]
    n_cols = min(len(col_lefts) - 1, innings + 2) + 1  # columns actually read
    return {"row_tops": row_tops, "row_bottoms": row_bottoms,
            "col_lefts": col_lefts[:n_cols], "cell_size": cell_size}


def main() -> int:
    current = {stem: _run(stem) for stem in BASELINE_GAMES + PHOTO_GAMES}
    if "--update" in sys.argv:
        # Baseline goldens are always the legacy geometry the verified reads came
        # from; only the photo goldens follow the current detector.
        golden = {stem: _run(stem, legacy=True) for stem in BASELINE_GAMES}
        golden.update({stem: current[stem] for stem in PHOTO_GAMES})
        GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
        print(f"Wrote {GOLDEN.name}")
        return 0

    golden = json.loads(GOLDEN.read_text())
    failures = 0
    for stem, cur in current.items():
        gold = golden[stem]
        tol = gold["cell_size"] * TOL_FRAC
        for key in ("row_tops", "row_bottoms", "col_lefts"):
            # How many columns get read depends on the game's --innings (its
            # _layout.json can appear/change after the golden was taken), so
            # compare the boundaries both have; only rows must match exactly.
            n = min(len(cur[key]), len(gold[key])) if key == "col_lefts" else len(gold[key])
            if len(cur[key]) < n or n < 2:
                print(f"FAIL {stem} {key}: {len(cur[key])} boundaries, golden has {len(gold[key])}")
                failures += 1
                continue
            cur[key], gold[key] = cur[key][:n], gold[key][:n]
            worst = max(abs(c - g) for c, g in zip(cur[key], gold[key]))
            status = "ok  " if worst <= tol else "FAIL"
            failures += worst > tol
            print(f"{status} {stem} {key}: max dev {worst}px (tol {tol:.1f}px)")
    print("\nall passed" if not failures else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
