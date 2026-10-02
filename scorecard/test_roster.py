"""Unit tests for the roster loader (#28): db.load_roster / parse_roster_file /
append_roster_player, on temp folders.

Run from scorecard/:  uv run python test_roster.py
No image or API needed.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import db


def _tmp_root() -> Path:
    d = Path(tempfile.mkdtemp(prefix="roster_test_"))
    return d


def test_comma_csv():
    root = _tmp_root()
    try:
        (root / "players.csv").write_text(
            "name,number,position,bats,throws,photo\n"
            "T. Kryston,,SS,R,R,t-kryston.jpg\n"
            "M. Gelaudi,98,C,R,R,\n",
            encoding="utf-8",
        )
        roster = db.load_roster(root)
        assert roster == [("T. Kryston", None), ("M. Gelaudi", "98")]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_semicolon_csv_with_bom():
    root = _tmp_root()
    try:
        text = (
            "name;number;position;bats;throws;photo\n"
            "G. Suares;2;2B;R;R;\n"
            "X. Dikkes;;OF;L;L;\n"
        )
        (root / "players.csv").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
        roster = db.load_roster(root)
        assert roster == [("G. Suares", "2"), ("X. Dikkes", None)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_txt_fallback_when_no_csv():
    root = _tmp_root()
    try:
        (root / "players.txt").write_text(
            "M. Gelaudi, 98\nT. Kryston, 0\n", encoding="utf-8",
        )
        roster = db.load_roster(root)
        # Legacy txt format: "0" is kept as a literal string (unknown-number
        # convention lives in the CSV migration, not the txt reader itself).
        assert roster == [("M. Gelaudi", "98"), ("T. Kryston", "0")]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_csv_preferred_over_txt_when_both_exist():
    root = _tmp_root()
    try:
        (root / "players.txt").write_text("Old Name, 1\n", encoding="utf-8")
        (root / "players.csv").write_text(
            "name,number,position,bats,throws,photo\nNew Name,2,,,,\n", encoding="utf-8",
        )
        roster = db.load_roster(root)
        assert roster == [("New Name", "2")]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_append_creates_csv_with_header():
    root = _tmp_root()
    try:
        path = db.append_roster_player(root, "New Player", 7)
        assert path == root / "players.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "name,number,position,bats,throws,photo"
        assert lines[1] == "New Player,7,,,,"
        assert db.load_roster(root) == [("New Player", "7")]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_append_migrates_existing_txt_roster():
    root = _tmp_root()
    try:
        (root / "players.txt").write_text(
            "M. Gelaudi, 98\nT. Kryston, 0\n", encoding="utf-8",
        )
        db.append_roster_player(root, "New Player", None)
        roster = db.load_roster(root)
        # Old roster carried over (number preserved, "0" preserved as-is from
        # txt), plus the new player appended with an empty number.
        assert roster == [("M. Gelaudi", "98"), ("T. Kryston", "0"), ("New Player", None)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_append_to_existing_csv_adds_one_row():
    root = _tmp_root()
    try:
        (root / "players.csv").write_text(
            "name,number,position,bats,throws,photo\nA,1,,,,\n", encoding="utf-8",
        )
        db.append_roster_player(root, "B", None)
        roster = db.load_roster(root)
        assert roster == [("A", "1"), ("B", None)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_append_guards_missing_trailing_newline():
    root = _tmp_root()
    try:
        # No trailing newline after the last row.
        (root / "players.csv").write_text(
            "name,number,position,bats,throws,photo\nA,1,,,,", encoding="utf-8",
        )
        db.append_roster_player(root, "B", None)
        text = (root / "players.csv").read_text(encoding="utf-8")
        # The new row must not have been glued onto the end of "A,1,,,,".
        assert "A,1,,,," in text.splitlines()
        roster = db.load_roster(root)
        assert roster == [("A", "1"), ("B", None)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_append_keeps_semicolon_delimiter():
    root = _tmp_root()
    try:
        (root / "players.csv").write_text(
            "name;number;position;bats;throws;photo\nA;1;;;;\n", encoding="utf-8",
        )
        db.append_roster_player(root, "B", 2)
        text = (root / "players.csv").read_text(encoding="utf-8")
        assert "B;2;;;;" in text.splitlines()
        roster = db.load_roster(root)
        assert roster == [("A", "1"), ("B", "2")]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_columns_are_fine():
    root = _tmp_root()
    try:
        (root / "players.csv").write_text("name\nSolo Player\n", encoding="utf-8")
        roster = db.load_roster(root)
        assert roster == [("Solo Player", None)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_roster_file_returns_empty():
    root = _tmp_root()
    try:
        assert db.load_roster(root) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
