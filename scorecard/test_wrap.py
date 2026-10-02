"""Unit tests for inning-wrap detection (#23): _resolve_col_to_inning.

Run from scorecard/:  uv run python test_wrap.py
No image or API needed — totals blankness is given directly.
"""
from __future__ import annotations

import extract_cells as ec

N_ROWS = 9


def _grid(cols: list[list[str | None]]):
    """cols[ci][ri] = result (None = empty); returns grid[ri][ci]."""
    n_cols = len(cols)
    g: list[list[dict | None]] = [[None] * n_cols for _ in range(N_ROWS)]
    for ci, col in enumerate(cols):
        for ri, res in enumerate(col):
            if res is not None:
                g[ri][ci] = {"result": res, "run": False}
    return g


E = [None] * N_ROWS
# Herons 2026-09-27: inning 4 batted around in column 4 (one non-out misread as
# K -> the grid sees 3 outs) and continued in column 5; column 8 is a complete
# inning in which one out was missed (grid sees 2 outs, all 9 batted).
HERONS = _grid([
    ["K", "E4", "F8"] + [None] * 6,
    [None, None, None, "K", "BB", "4-3", "6-3", None, None],
    ["E6", "6-3", None, None, None, None, None, "K", "K"],
    ["1B", "1B", "2B", "E", "4-3", "1B", "BB", "K", "K"],
    [None, None, "1B", "1B", "4-3", None, None, None, None],
    ["4-3", "E6", None, None, None, "1B", "1B", "2B", "F7"],
    [None, None, "1B", "F6", "6-3", None, None, None, None],
    ["1B", "2B", "1B", "BB", "FC", "2B", "BB", "K", "F7"],
    [None, None, None, None, None, "K", "F5", None, None],
    E,
])
HERONS_TRUE = [1, 2, 3, 4, 4, 5, 6, 7, 8, 9]
# Totals row: written everywhere a full inning ended; column 4 (wrap) and the
# unused column 10 blank.
HERONS_BLANK = [False, False, False, True, False, False, False, False, False, True]


def test_grid_alone_gets_herons_wrong():
    c2i, src = ec._resolve_col_to_inning(HERONS, N_ROWS, 10)
    assert src == "grid"
    assert c2i != HERONS_TRUE


def test_blank_totals_find_wrap_and_veto_false_one():
    c2i, src = ec._resolve_col_to_inning(HERONS, N_ROWS, 10, totals_blank=HERONS_BLANK)
    assert (c2i, src) == (HERONS_TRUE, "totals")


def test_sparse_totals_row_is_ignored():
    """Grizzlies 2026-05-10: scorekeeper left most totals blank -> not evidence."""
    blank = [True] * 10
    blank[4] = False
    c2i, src = ec._resolve_col_to_inning(HERONS, N_ROWS, 10, totals_blank=blank)
    assert src == "grid"


def test_blank_totals_under_short_column_is_not_a_wrap():
    """A forgotten totals cell under a 3-batter inning does not make a wrap."""
    blank = list(HERONS_BLANK)
    blank[0] = True  # column 1 has only 3 PAs
    c2i, src = ec._resolve_col_to_inning(HERONS, N_ROWS, 10, totals_blank=blank)
    assert (c2i, src) == (HERONS_TRUE, "totals")


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
