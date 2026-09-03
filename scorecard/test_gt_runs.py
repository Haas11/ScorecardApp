"""Unit tests for _reconcile_gt_runs (#1: extracted runs == GT runs per inning).

Run from scorecard/:  uv run python test_gt_runs.py
No image or API needed — the VLM re-check is a fake callable.
"""
from __future__ import annotations

import extract_cells as ec


def _grid(n_rows: int, n_cols: int, cells: dict[tuple[int, int], dict]):
    g: list[list[dict | None]] = [[None] * n_cols for _ in range(n_rows)]
    for (ri, ci), c in cells.items():
        g[ri][ci] = dict(c)
    return g


def _runs(grid, ci: int = 0) -> list[int]:
    """0-based rows with run=True in column ci."""
    return [ri for ri, row in enumerate(grid) if (row[ci] or {}).get("run")]


class FakeRecheck:
    def __init__(self, truth: dict[int, bool]):
        self.truth = truth           # ri -> answer
        self.calls: list[int] = []

    def __call__(self, ri: int, ci: int, inn: int) -> bool:
        self.calls.append(ri)
        return self.truth.get(ri, False)


def _reconcile(grid, gt_r: int, recheck=None, last_batter=None, inn: int = 1, n_rows: int | None = None):
    n = n_rows or len(grid)
    return ec._reconcile_gt_runs(
        grid, n, {inn: [0]}, {inn: {"R": gt_r, "H": 0, "E": 0, "LOB": 0}},
        last_batter or {}, recheck, None,
    )


# ── under-count: structural tier ──────────────────────────────────────────────

def test_runners_ahead_of_scorer_forced_when_all_outs_are_pa_outs():
    # P1 1B, P2 BB (both no run), P3 2B run, P4-P6 out.  GT R=3 → P1, P2 must have scored.
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": False, "run_conf": 5},
        (1, 0): {"result": "BB", "run": False, "run_conf": 5},
        (2, 0): {"result": "2B", "run": True, "rbi_slot": 9},
        (3, 0): {"result": "K", "run": False},
        (4, 0): {"result": "F7", "run": False},
        (5, 0): {"result": "6-3", "run": False},
    })
    rc = FakeRecheck({})
    msgs = _reconcile(grid, 3, rc)
    assert _runs(grid) == [0, 1, 2]
    assert rc.calls == [], "forced runs must not spend a VLM call"
    assert "gt:run->True(forced)" in grid[0][0]["adjusted"]
    assert any("forced True" in m for m in msgs)


def test_runner_behind_last_scorer_is_not_forced():
    # P4 reached AFTER the last scorer → could be LOB; not forced, goes to re-check.
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": True},
        (1, 0): {"result": "K", "run": False},
        (2, 0): {"result": "K", "run": False},
        (3, 0): {"result": "BB", "run": False},
        (4, 0): {"result": "F7", "run": False},
    })
    rc = FakeRecheck({3: False})
    msgs = _reconcile(grid, 2, rc)
    assert _runs(grid) == [0]
    assert rc.calls == [3]
    assert any("left for review" in m for m in msgs)


def test_more_forced_than_missing_assigns_nothing():
    # Two runners ahead of the scorer but GT says only one run is missing → a
    # run=True cell is probably the wrong one; do not fabricate.
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": False},
        (1, 0): {"result": "BB", "run": False},
        (2, 0): {"result": "2B", "run": True},
        (3, 0): {"result": "K", "run": False},
        (4, 0): {"result": "F7", "run": False},
        (5, 0): {"result": "6-3", "run": False},
    })
    rc = FakeRecheck({})
    msgs = _reconcile(grid, 2, rc)
    assert _runs(grid) == [2]
    assert any("probably wrong" in m for m in msgs)


def test_dp_disables_structural_tier():
    # With a DP one out may have been a baserunner → runner ahead may be retired.
    grid = _grid(5, 1, {
        (0, 0): {"result": "1B", "run": False, "run_conf": 5},
        (1, 0): {"result": "2B", "run": True},
        (2, 0): {"result": "DP", "run": False},
        (3, 0): {"result": "K", "run": False},
        (4, 0): {"result": "K", "run": False},
    })
    rc = FakeRecheck({0: True})
    _reconcile(grid, 2, rc)
    assert rc.calls == [0], "must go through VLM re-check, not forced"
    assert _runs(grid) == [0, 1]
    assert "recheck:run->True" in grid[0][0]["adjusted"]


def test_runner_out_note_excludes_from_forced():
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": False, "notes": "runner out 2-6 in top-right quadrant"},
        (1, 0): {"result": "BB", "run": False},
        (2, 0): {"result": "2B", "run": True},
        (3, 0): {"result": "K", "run": False},
        (4, 0): {"result": "F7", "run": False},
        (5, 0): {"result": "6-3", "run": False},
    })
    rc = FakeRecheck({0: False})
    _reconcile(grid, 3, rc)
    assert grid[1][0]["run"] is True          # forced
    assert grid[0][0]["run"] is False         # excluded, re-check said no
    assert rc.calls == [0]


def test_only_two_pa_outs_is_not_airtight():
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": False, "run_conf": 4},
        (1, 0): {"result": "2B", "run": True},
        (2, 0): {"result": "K", "run": False},
        (3, 0): {"result": "K", "run": False},
    })
    rc = FakeRecheck({0: False})
    _reconcile(grid, 2, rc)
    assert rc.calls == [0]
    assert _runs(grid) == [1]


# ── under-count: ranked re-check tier ─────────────────────────────────────────

def test_recheck_order_is_lowest_run_conf_then_sequence():
    grid = _grid(4, 1, {
        (0, 0): {"result": "1B", "run": False, "run_conf": 5},
        (1, 0): {"result": "BB", "run": False, "run_conf": 2},
        (2, 0): {"result": "E6", "run": False, "run_conf": 5},
        (3, 0): {"result": "K", "run": False},
    })
    rc = FakeRecheck({1: True})
    _reconcile(grid, 1, rc)
    assert rc.calls == [1], rc.calls          # lowest run_conf first, stop when filled
    assert _runs(grid) == [1]


def test_recheck_all_false_leaves_grid_and_flags_review():
    grid = _grid(3, 1, {
        (0, 0): {"result": "1B", "run": False},
        (1, 0): {"result": "BB", "run": False},
        (2, 0): {"result": "K", "run": False},
    })
    rc = FakeRecheck({})
    msgs = _reconcile(grid, 1, rc)
    assert _runs(grid) == []
    assert rc.calls == [0, 1]
    assert any("still 1 run(s) short" in m for m in msgs)


def test_no_recheck_callable_skips_tier_two():
    grid = _grid(2, 1, {(0, 0): {"result": "1B", "run": False}})
    msgs = _reconcile(grid, 1, None)
    assert _runs(grid) == []
    assert any("left for review" in m for m in msgs)


def test_matching_inning_is_untouched():
    grid = _grid(2, 1, {(0, 0): {"result": "1B", "run": True}, (1, 0): {"result": "K", "run": False}})
    rc = FakeRecheck({1: True})
    msgs = _reconcile(grid, 1, rc)
    assert msgs == [] and rc.calls == [] and _runs(grid) == [0]


# ── cyclic batting order ──────────────────────────────────────────────────────

def test_sequence_uses_previous_innings_last_batter():
    # Inning 2 starts at slot 8 (row 7) because inning 1 ended with slot 7.
    # Row 7 BB (ahead of scorer row 8) is forced; row 0 1B (behind) is not.
    n = 9
    cells = {
        (7, 0): {"result": "BB", "run": False},
        (8, 0): {"result": "1B", "run": True},
        (0, 0): {"result": "1B", "run": False},
        (1, 0): {"result": "K", "run": False},
        (2, 0): {"result": "F8", "run": False},
        (3, 0): {"result": "4-3", "run": False},
    }
    grid = _grid(n, 1, cells)
    rc = FakeRecheck({0: False})
    _reconcile(grid, 2, rc, last_batter={1: 7}, inn=2)
    assert grid[7][0]["run"] is True          # forced: ahead of the scorer in sequence
    assert grid[0][0]["run"] is False         # behind the scorer: not forced
    assert rc.calls == []                     # the forced run filled the shortfall


def test_unknown_start_disables_structural_tier():
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": False},
        (1, 0): {"result": "2B", "run": True},
        (2, 0): {"result": "K", "run": False},
        (3, 0): {"result": "K", "run": False},
        (4, 0): {"result": "K", "run": False},
    })
    rc = FakeRecheck({0: True})
    _reconcile(grid, 2, rc, last_batter={}, inn=3)   # inning 2's last batter unknown
    assert rc.calls == [0]


# ── over-count ────────────────────────────────────────────────────────────────

def test_overcount_removes_lowest_run_conf_with_clear_gap():
    grid = _grid(4, 1, {
        (0, 0): {"result": "1B", "run": True, "run_conf": 5, "rbi_slot": 3},
        (1, 0): {"result": "BB", "run": True, "run_conf": 2},          # no rbi digit → 1
        (2, 0): {"result": "K", "run": False},
        (3, 0): {"result": "K", "run": False},
    })
    msgs = _reconcile(grid, 1, None)
    assert _runs(grid) == [0]
    assert "gt:run->False" in grid[1][0]["adjusted"]
    assert any("run removed" in m for m in msgs)


def test_overcount_without_gap_removes_nothing():
    grid = _grid(4, 1, {
        (0, 0): {"result": "1B", "run": True, "run_conf": 4, "rbi_slot": 3},
        (1, 0): {"result": "BB", "run": True, "run_conf": 4, "rbi_slot": 3},
        (2, 0): {"result": "K", "run": False},
        (3, 0): {"result": "K", "run": False},
    })
    msgs = _reconcile(grid, 1, None)
    assert _runs(grid) == [0, 1]
    assert any("no clear confidence gap" in m for m in msgs)


def test_overcount_hr_is_never_removed_and_sole_other_scorer_goes():
    grid = _grid(3, 1, {
        (0, 0): {"result": "HR", "run": True, "run_conf": 5},
        (1, 0): {"result": "1B", "run": True, "run_conf": 5, "rbi_slot": 1},
        (2, 0): {"result": "K", "run": False},
    })
    _reconcile(grid, 1, None)
    assert _runs(grid) == [0]


def test_overcount_more_hr_than_gt_flags_contradiction():
    grid = _grid(2, 1, {
        (0, 0): {"result": "HR", "run": True},
        (1, 0): {"result": "HR", "run": True},
    })
    msgs = _reconcile(grid, 1, None)
    assert _runs(grid) == [0, 1]
    assert any("contradicts" in m for m in msgs)


def test_overcount_scorer_behind_nonscorer_is_structurally_suspect():
    # Both scorers self-report 5; P3 scored while P2 (ahead) did not, in an
    # airtight inning → P3 gets the structural dock and is the one removed.
    grid = _grid(6, 1, {
        (0, 0): {"result": "1B", "run": True, "run_conf": 5, "rbi_slot": 4},
        (1, 0): {"result": "BB", "run": False, "run_conf": 5},
        (2, 0): {"result": "1B", "run": True, "run_conf": 5, "rbi_slot": 4},
        (3, 0): {"result": "K", "run": False},
        (4, 0): {"result": "K", "run": False},
        (5, 0): {"result": "K", "run": False},
    })
    msgs = _reconcile(grid, 1, None)
    # keys: P1 = 5, P3 = 4 → gap 1 < 2 → nothing removed, but P3 listed first
    assert _runs(grid) == [0, 2]
    review = [m for m in msgs if "no clear confidence gap" in m][0]
    assert review.index("P3") < review.index("P1")


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
