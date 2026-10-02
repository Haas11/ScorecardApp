"""Unit tests for data.py's roster CSV parsing (#28): data._parse_roster_csv.

Run from viewer/:  uv run --project ../scorecard python test_data.py
No workbook, Drive secrets, or network access needed -- pure string parsing.
"""
from __future__ import annotations

import data


def test_comma_csv():
    text = (
        "name,number,position,bats,throws,photo\n"
        "T. Kryston,,SS,R,R,t-kryston.jpg\n"
        "M. Gelaudi,98,C,R,R,\n"
    )
    df = data._parse_roster_csv(text)
    assert list(df.columns) == data._ROSTER_COLUMNS
    assert df.loc[0, "name"] == "T. Kryston"
    assert df.loc[0, "number"] == ""
    assert df.loc[0, "photo"] == "t-kryston.jpg"
    assert df.loc[1, "number"] == "98"


def test_semicolon_csv_with_bom():
    # BOM stripping happens via utf-8-sig decode before _parse_roster_csv is
    # called (roster() does that); here we just check ';' delimiter sniffing.
    text = (
        "name;number;position;bats;throws;photo\n"
        "G. Suares;2;2B;R;R;\n"
    )
    df = data._parse_roster_csv(text)
    assert df.loc[0, "name"] == "G. Suares"
    assert df.loc[0, "number"] == "2"
    assert df.loc[0, "position"] == "2B"


def test_missing_columns_are_fine():
    df = data._parse_roster_csv("name\nSolo Player\n")
    assert list(df.columns) == data._ROSTER_COLUMNS
    assert df.loc[0, "name"] == "Solo Player"
    assert df.loc[0, "number"] == ""


def test_empty_text_returns_empty_frame_with_columns():
    df = data._parse_roster_csv("")
    assert list(df.columns) == data._ROSTER_COLUMNS
    assert len(df) == 0


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
