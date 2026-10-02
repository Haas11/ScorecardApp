"""Unit tests for players.py: rating scale (#24) and cumulative AVG.

Run from viewer/:  uv run --project ../scorecard python test_players.py
No workbook needed -- small hand-made DataFrames.
"""
from __future__ import annotations

import pandas as pd

import players


def _players(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_best_median_worst():
    df = _players([
        {"Name": "Worst", "AVG": 0.100, "eligible": True},
        {"Name": "Median", "AVG": 0.200, "eligible": True},
        {"Name": "Best", "AVG": 0.300, "eligible": True},
    ])
    r = players.ratings(df)
    assert r.loc[0, "AVG"] == 1
    assert r.loc[1, "AVG"] == 50
    assert r.loc[2, "AVG"] == 100


def test_ties_share_average_rank():
    df = _players([
        {"Name": "A", "AVG": 0.200, "eligible": True},
        {"Name": "B", "AVG": 0.200, "eligible": True},
        {"Name": "C", "AVG": 0.300, "eligible": True},
    ])
    r = players.ratings(df)
    # A and B tie for ranks 1 and 2 -> average rank 1.5 -> same rating, between worst and best
    assert r.loc[0, "AVG"] == r.loc[1, "AVG"]
    assert 1 < r.loc[0, "AVG"] < r.loc[2, "AVG"]
    assert r.loc[2, "AVG"] == 100


def test_k_pct_inverted():
    df = _players([
        {"Name": "LowK", "K%": 0.05, "eligible": True},
        {"Name": "HighK", "K%": 0.30, "eligible": True},
    ])
    r = players.ratings(df)
    assert r.loc[0, "K%"] == 100  # lowest K% is best
    assert r.loc[1, "K%"] == 1


def test_non_eligible_gets_na():
    df = _players([
        {"Name": "Eligible", "AVG": 0.300, "eligible": True},
        {"Name": "SmallSample", "AVG": 0.900, "eligible": False},
    ])
    r = players.ratings(df)
    assert r.loc[0, "AVG"] == 100
    assert pd.isna(r.loc[1, "AVG"])


def test_missing_value_gets_na():
    df = _players([
        {"Name": "A", "AVG": 0.300, "BABIP": 0.250, "eligible": True},
        {"Name": "B", "AVG": 0.200, "BABIP": None, "eligible": True},
    ])
    r = players.ratings(df)
    assert r.loc[0, "BABIP"] == 100
    assert pd.isna(r.loc[1, "BABIP"])


def test_single_eligible_value_gets_100():
    df = _players([
        {"Name": "Solo", "AVG": 0.250, "eligible": True},
        {"Name": "Small", "AVG": 0.900, "eligible": False},
    ])
    r = players.ratings(df)
    assert r.loc[0, "AVG"] == 100


def test_cumulative_avg_with_zero_ab_game():
    games = pd.DataFrame([
        {"Name": "P", "Date": "2026-04-01", "AB": 3, "H": 1},
        {"Name": "P", "Date": "2026-04-08", "AB": 0, "H": 0},  # walk-only game
        {"Name": "P", "Date": "2026-04-15", "AB": 2, "H": 2},
    ])
    games["Date"] = pd.to_datetime(games["Date"])
    out = players.player_games(games, "P")
    assert list(out["cum_AB"]) == [3, 3, 5]
    assert list(out["cum_H"]) == [1, 1, 3]
    assert out.loc[0, "cum_AVG"] == 1 / 3
    assert out.loc[1, "cum_AVG"] == 1 / 3  # AB unchanged, not NaN
    assert out.loc[2, "cum_AVG"] == 3 / 5


def test_cumulative_avg_nan_while_no_ab_yet():
    games = pd.DataFrame([
        {"Name": "P", "Date": "2026-04-01", "AB": 0, "H": 0},
        {"Name": "P", "Date": "2026-04-08", "AB": 2, "H": 1},
    ])
    games["Date"] = pd.to_datetime(games["Date"])
    out = players.player_games(games, "P")
    assert pd.isna(out.loc[0, "cum_AVG"])
    assert out.loc[1, "cum_AVG"] == 1 / 2


def test_slug():
    assert players.slug("T. Kryston") == "t-kryston"
    assert players.slug("G.L. McLelland") == "g-l-mclelland"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failed else 0)
