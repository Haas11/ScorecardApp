"""Unit tests for _score_cell and its wiring into _build_slot_data (#11).

Run from scorecard/:  uv run python -m pytest test_confidence.py -q
(or plain:            uv run python test_confidence.py)
"""
from __future__ import annotations

import extract_cells as ec


def test_full_self_report_no_adjustment():
    cell = {"result": "1B", "run": False, "result_conf": 5, "run_conf": 5}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf, reasons) == (5, 5, [])


def test_missing_self_report_is_legacy_four():
    cell = {"result": "1B", "run": False}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf) == (4, 4)
    assert reasons == ["legacy"]


def test_reread_docks_both_by_one():
    cell = {"result": "1B", "run": False, "result_conf": 5, "run_conf": 5, "reread": "hole"}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf) == (4, 4)
    assert reasons == ["reread:hole"]


def test_each_adjusted_flag_docks_run_conf_by_one():
    cell = {
        "result": "K", "run": False, "result_conf": 5, "run_conf": 5,
        "adjusted": ["constraint:out->run=False", "gt:run->False"],
    }
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert result_conf == 5
    assert run_conf == 3
    assert reasons == ["constraint:out->run=False", "gt:run->False"]


def test_run_true_without_rbi_digit_docks_run_conf():
    cell = {"result": "BB", "run": True, "result_conf": 5, "run_conf": 5, "rbi_slot": None}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf) == (5, 4)
    assert reasons == ["run_without_rbi_digit"]


def test_run_true_with_rbi_digit_is_not_docked():
    cell = {"result": "BB", "run": True, "result_conf": 5, "run_conf": 5, "rbi_slot": 3}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf, reasons) == (5, 5, [])


def test_inning_r_mismatch_docks_run_conf_only():
    cell = {"result": "K", "run": False, "result_conf": 5, "run_conf": 5}
    result_conf, run_conf, reasons = ec._score_cell(cell, 3, {3: {"R"}})
    assert (result_conf, run_conf) == (5, 4)
    assert reasons == ["inning_R_mismatch"]


def test_inning_h_mismatch_only_docks_result_conf_for_hits():
    hit_cell = {"result": "1B", "run": False, "result_conf": 5, "run_conf": 5}
    result_conf, run_conf, reasons = ec._score_cell(hit_cell, 2, {2: {"H"}})
    assert (result_conf, run_conf) == (4, 5)
    assert reasons == ["inning_H_mismatch"]

    out_cell = {"result": "K", "run": False, "result_conf": 5, "run_conf": 5}
    result_conf2, run_conf2, reasons2 = ec._score_cell(out_cell, 2, {2: {"H"}})
    assert (result_conf2, run_conf2, reasons2) == (5, 5, [])


def test_ambiguous_notes_dock_result_conf():
    cell = {"result": "K", "run": False, "result_conf": 5, "run_conf": 5,
            "notes": "Ambiguous out mark ... defaulting to K"}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf) == (4, 5)
    assert reasons == ["ambiguous_notes"]


def test_floors_at_one_never_goes_negative():
    cell = {
        "result": "K", "run": True, "result_conf": 1, "run_conf": 1,
        "reread": "run", "rbi_slot": None,
        "adjusted": ["constraint:out->run=False", "gt:run->False", "recheck:run->True"],
    }
    result_conf, run_conf, _ = ec._score_cell(cell, 1, {1: {"R"}})
    assert result_conf == 1
    assert run_conf == 1


def test_build_slot_data_propagates_confidence_onto_pa():
    grid = [[{"result": "1B", "run": False, "result_conf": 3, "run_conf": 5}]]
    slot_info = [{"starter": ("P1", None), "subs": []}]
    slot_data, _ = ec._build_slot_data(grid, slot_info, 1, 1, [1])
    _, pa_lists = slot_data[0]
    pa = pa_lists[0][0]
    assert pa.result_conf == 3
    assert pa.run_conf == 5
    assert pa.confidence == 3  # min(result_conf, run_conf)
    assert pa.conf_reasons == []


def test_build_slot_data_uses_inning_flags_when_given():
    grid = [[{"result": "1B", "run": False, "result_conf": 5, "run_conf": 5}]]
    slot_info = [{"starter": ("P1", None), "subs": []}]
    slot_data, _ = ec._build_slot_data(grid, slot_info, 1, 1, [4], inning_flags={4: {"H"}})
    _, pa_lists = slot_data[0]
    pa = pa_lists[0][0]
    assert pa.result_conf == 4
    assert pa.conf_reasons == ["inning_H_mismatch"]


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
