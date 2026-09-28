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


def test_circle_mismatch_docks_result_not_run():
    cell = {"result": "1B", "run": False, "result_conf": 5, "run_conf": 5, "adjusted": ["circle_mismatch"]}
    result_conf, run_conf, reasons = ec._score_cell(cell, 1, None)
    assert (result_conf, run_conf, reasons) == (3, 5, ["circle_mismatch"])


def test_circle_reread_docks_result_once():
    cell = {"result": "F8", "run": False, "result_conf": 5, "run_conf": 5, "adjusted": ["circle:1B->F8"]}
    assert ec._score_cell(cell, 1, None)[:2] == (4, 5)


def test_slot_mismatch_docks_ab_cells_only():
    walk = {"result": "BB", "run": False, "result_conf": 5, "run_conf": 5}
    out = {"result": "K", "run": False, "result_conf": 5, "run_conf": 5}
    assert ec._score_cell(walk, 1, None, {"H"})[0] == 5      # BB is not an at-bat
    rc, _, reasons = ec._score_cell(out, 1, None, {"H"})
    assert (rc, reasons) == (4, ["slot_H_mismatch"])


def test_row_and_inning_h_mismatch_crossing_docks_twice():
    cell = {"result": "K", "run": False, "result_conf": 5, "run_conf": 5}
    rc, _, reasons = ec._score_cell(cell, 2, {2: {"H"}}, {"H"})
    assert rc == 3 and "row_and_inning_H_mismatch" in reasons


def test_build_slot_data_passes_row_flags():
    grid = [[{"result": "F7", "run": False, "result_conf": 5, "run_conf": 5}]]
    slot_info = [{"starter": ("A", 1), "subs": []}]
    slot_data, _ = ec._build_slot_data(grid, slot_info, 1, 1, [1], row_flags={1: {"AB"}})
    _, pa_lists = slot_data[0]
    pa = pa_lists[0][0]
    assert pa.result_conf == 4 and pa.conf_reasons == ["slot_AB_mismatch"]


def test_parse_avg_formats():
    assert ec._parse_avg(".500") == 0.5
    assert ec._parse_avg("250") == 0.25
    assert ec._parse_avg("1.000") == 1.0
    assert ec._parse_avg("1") == 1.0
    assert ec._parse_avg("0") == 0.0
    assert ec._parse_avg("") is None


def test_stats_line_must_match_avg():
    assert ec._stats_line_ok(2, 3, 0.667) and ec._stats_line_ok(2, 3, 0.666)
    assert ec._stats_line_ok(0, 4, 0.0)
    assert not ec._stats_line_ok(2, 4, 0.250)   # misread: 2-4 is .500
    assert not ec._stats_line_ok(5, 4, 1.0)     # H > AB impossible
    assert not ec._stats_line_ok(1, 4, None)    # no AVG to confirm


def test_stat_class_ignores_fielder_digits_only():
    assert ec._stat_class("F7") == ec._stat_class("F8") == "FLY"
    assert ec._stat_class("6-3") == ec._stat_class("4-3") == "GROUND"
    assert ec._stat_class("E") == ec._stat_class("E53") == "E"
    assert ec._stat_class("F8") != ec._stat_class("K")
    assert ec._stat_class("2B") != ec._stat_class("3B")
    assert ec._stat_class("HBP") == ec._stat_class("HP")


def test_verify_disagreement_docks_and_names_alternative():
    cell = {"result": "K", "run": False, "result_conf": 5, "run_conf": 5,
            "verify": {"model": "m", "result": "F8", "run": False}}
    rc, uc, reasons = ec._score_cell(cell, 1, None)
    assert (rc, uc) == (3, 5) and reasons == ["verify:result K|F8"]


def test_verify_agreement_bumps_result_conf():
    cell = {"result": "F7", "run": False, "result_conf": 3, "run_conf": 5,
            "verify": {"model": "m", "result": "F8", "run": False}}
    rc, _, reasons = ec._score_cell(cell, 1, None)
    assert rc == 4 and reasons == ["verify:agree"]


def test_verify_verdict_follows_hand_correction():
    # First read K, second read F8; the user corrects the cell to F8 -> now agrees.
    cell = {"result": "F8", "run": False, "result_conf": 5, "run_conf": 5,
            "verify": {"model": "m", "result": "F8", "run": False}}
    assert ec._verify_agreement(cell) == (True, True)


def test_verify_run_not_compared_after_a_pass_changed_it():
    cell = {"result": "1B", "run": True, "adjusted": ["gt:run->True(forced)"],
            "verify": {"model": "m", "result": "1B", "run": False}}
    assert ec._verify_agreement(cell) == (True, None)


def test_verify_candidates_selection():
    grid = [[
        {"result": "1B", "run": False, "result_conf": 2, "run_conf": 5},            # low conf
        {"result": "2B", "run": True, "result_conf": 5, "run_conf": 5},             # confident: skip
        {"result": "K", "run": False},                                              # legacy: skip
        {"result": "F8", "run": False, "adjusted": ["circle_mismatch"]},            # circle
        {"result": "BB", "run": False, "result_conf": 2, "run_conf": 2,
         "verify": {"model": "m"}},                                                 # already done
        {"result": "6-3", "run": False, "result_conf": 5, "run_conf": 5},           # H mismatch inning
        None,
    ]]
    cands = ec._verify_candidates(grid, 1, 7, [1, 1, 1, 1, 1, 2, 2], 9, {2: {"H"}}, "m")
    assert [(ci, why) for _, ci, why in cands] == [
        (0, "conf 2/5"), (3, "circle_mismatch"), (5, "inning_H_mismatch")]


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
