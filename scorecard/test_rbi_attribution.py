"""Unit tests for _build_slot_data (PA construction + RBI attribution).

Run from scorecard/:  uv run python -m pytest test_rbi_attribution.py -q
(or plain:            uv run python test_rbi_attribution.py)

Regression for improvement #12: a 2-run HR was credited with 3 RBI when the VLM
read the batter's own slot digit in the HR cell's bottom-left quadrant.
"""
from __future__ import annotations

import numpy as np

import extract_cells as ec


def _slot_info(n: int, subs: dict[int, list[tuple[tuple[str, int | None], int]]] | None = None):
    subs = subs or {}
    return [
        {"starter": (f"P{ri + 1}", None), "subs": subs.get(ri, [])}
        for ri in range(n)
    ]


def _grid(n_rows: int, n_cols: int, cells: dict[tuple[int, int], dict]):
    g: list[list[dict | None]] = [[None] * n_cols for _ in range(n_rows)]
    for (ri, ci), c in cells.items():
        g[ri][ci] = c
    return g


def _rbi(slot_data, ri: int, player_idx: int = 0) -> list[tuple[int, str, int]]:
    _, pa_lists = slot_data[ri]
    return [(pa.inning, pa.result, pa.rbi) for pa in pa_lists[player_idx]]


def test_two_run_hr_with_self_digit_is_two_rbi_not_three():
    # Slot 2 singles and scores on slot 3's HR (rbi_slot=3 on slot 2's cell).
    # The VLM also read "3" in the HR cell's own bottom-left (the scorer wrote
    # the batter's own number there).  Correct total for slot 3 = 2 RBI.
    grid = _grid(3, 1, {
        (1, 0): {"result": "1B", "run": True, "rbi_slot": 3},
        (2, 0): {"result": "HR", "run": True, "rbi_slot": 3},
    })
    slot_data, warns = ec._build_slot_data(grid, _slot_info(3), 3, 1, [1])
    assert _rbi(slot_data, 2) == [(1, "HR", 2)], _rbi(slot_data, 2)
    assert warns == []


def test_two_run_hr_with_null_self_digit_is_two_rbi():
    # Same as above but the HR cell's rbi_slot was read as null — must give the
    # same answer (this is the "clean slate" run that hid the bug).
    grid = _grid(3, 1, {
        (1, 0): {"result": "1B", "run": True, "rbi_slot": 3},
        (2, 0): {"result": "HR", "run": True, "rbi_slot": None},
    })
    slot_data, _ = ec._build_slot_data(grid, _slot_info(3), 3, 1, [1])
    assert _rbi(slot_data, 2) == [(1, "HR", 2)]


def test_solo_hr_is_one_rbi_regardless_of_self_digit():
    for self_digit in (None, 1):
        grid = _grid(1, 1, {(0, 0): {"result": "HR", "run": True, "rbi_slot": self_digit}})
        slot_data, _ = ec._build_slot_data(grid, _slot_info(1), 1, 1, [1])
        assert _rbi(slot_data, 0) == [(1, "HR", 1)]


def test_self_digit_on_non_hr_is_not_credited_and_warns():
    grid = _grid(2, 1, {
        (0, 0): {"result": "1B", "run": True, "rbi_slot": 1},   # impossible
    })
    slot_data, warns = ec._build_slot_data(grid, _slot_info(2), 2, 1, [1])
    assert _rbi(slot_data, 0) == [(1, "1B", 0)]
    assert len(warns) == 1 and "impossible" in warns[0]


def test_rbi_credited_to_correct_pa_and_sub():
    # Slot 1: starter until inning 3, sub from inning 4.  Slot 2 scores in
    # inning 2 (starter gets the RBI) and inning 5 (sub gets the RBI).
    subs = {0: [(("Sub1", None), 4)]}
    grid = _grid(2, 5, {
        (0, 1): {"result": "2B", "run": False},
        (0, 4): {"result": "1B", "run": False},
        (1, 1): {"result": "BB", "run": True, "rbi_slot": 1},
        (1, 4): {"result": "BB", "run": True, "rbi_slot": 1},
    })
    slot_data, _ = ec._build_slot_data(grid, _slot_info(2, subs), 2, 5, [1, 2, 3, 4, 5])
    assert _rbi(slot_data, 0, 0) == [(2, "2B", 1)]
    assert _rbi(slot_data, 0, 1) == [(5, "1B", 1)]


def test_run_without_result_never_credits():
    grid = _grid(2, 1, {
        (0, 0): {"result": "1B", "run": False},
        (1, 0): {"result": None, "run": True, "rbi_slot": 1},   # dropped cell
    })
    slot_data, _ = ec._build_slot_data(grid, _slot_info(2), 2, 1, [1])
    assert _rbi(slot_data, 0) == [(1, "1B", 0)]


# ── #7: RBI <= Runs per inning invariant ──────────────────────────────────────

def test_check_rbi_leq_runs_no_violation():
    slot_data = [
        ([(("P1", None), 0)], [[ec.PlateAppearance(inning=1, result="HR", run_scored=True, rbi=1)]]),
    ]
    assert ec._check_rbi_leq_runs(slot_data) == []


def test_check_rbi_leq_runs_flags_violation():
    slot_data = [
        ([(("P1", None), 0)], [[ec.PlateAppearance(inning=1, result="K", run_scored=False, rbi=1)]]),
    ]
    warnings = ec._check_rbi_leq_runs(slot_data)
    assert len(warnings) == 1
    assert "inn 1" in warnings[0] and "RBI total 1 > runs scored 0" in warnings[0]


def test_classify_cell_omits_rbi_slot_key_when_run_false():
    # #7: a run=False cell must not carry "rbi_slot": None — that would look
    # identical to "read attempted, no digit found" and block a later pass
    # (hole-reread, HR constraint, GT reconciliation) that flips run to True
    # from ever triggering _backfill_rbi_cells for this cell.
    orig_call_api = ec._call_api
    ec._call_api = lambda *a, **k: '{"result": "K", "run": false, "result_conf": 5, "run_conf": 5}'
    try:
        result = ec.classify_cell(np.zeros((10, 10, 3), dtype=np.uint8), "P1", 1, client=None, model="x")
    finally:
        ec._call_api = orig_call_api
    assert result.get("run") is False
    assert "rbi_slot" not in result


def test_classify_cell_sets_rbi_slot_key_when_run_true():
    orig_call_api, orig_read_rbi = ec._call_api, ec._read_rbi_slot
    ec._call_api = lambda *a, **k: '{"result": "1B", "run": true, "result_conf": 5, "run_conf": 5}'
    ec._read_rbi_slot = lambda *a, **k: 4
    try:
        result = ec.classify_cell(np.zeros((10, 10, 3), dtype=np.uint8), "P1", 1, client=None, model="x")
    finally:
        ec._call_api, ec._read_rbi_slot = orig_call_api, orig_read_rbi
    assert result.get("rbi_slot") == 4


def test_build_slot_data_fallback_credit_can_trigger_rbi_leq_runs_warning():
    # P1's only PA is in inning 2 with no run. A run cell in inning 1 credits
    # rbi_slot=1 (P1's slot), but P1 has no PA in inning 1 to attach it to, so
    # the fallback ("credit the batter's last PA") lands the RBI on the
    # inning-2 PA instead — _check_rbi_leq_runs should flag that mismatch.
    grid = _grid(2, 2, {
        (0, 1): {"result": "K", "run": False},                # P1's only PA: inn 2, no run
        (1, 0): {"result": "BB", "run": True, "rbi_slot": 1},  # inn 1: runner scores, credits slot 1
    })
    slot_data, warnings = ec._build_slot_data(grid, _slot_info(2), 2, 2, [1, 2])
    assert _rbi(slot_data, 0) == [(2, "K", 1)]
    assert any("inn 2" in w and "RBI total 1 > runs scored 0" in w for w in warnings)


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
    sys.exit(1 if failed else 0)
