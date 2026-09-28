"""Unit tests for gamepaths.resolve_game (#20). No image or API needed.

Run from scorecard/:  uv run python test_gamepaths.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from gamepaths import GameNotFound, cells_json, find_scan, resolve_game

NAME = "2026-07-12 Grizzlies (Home)"


def _layout(tmp: Path) -> Path:
    root = tmp / "Quick 2026"
    g = root / "games" / NAME
    (g / "cells").mkdir(parents=True)
    (g / f"{NAME}_cells.json").write_text("{}")
    (g / "cells" / "r01_c01.json").write_text("{}")
    (root / "scans").mkdir()
    (root / "scans" / f"{NAME}.jpg").write_bytes(b"")
    return root


def test_all_forms_resolve_to_the_same_game():
    with tempfile.TemporaryDirectory() as t:
        root = _layout(Path(t)).resolve()
        g = root / "games" / NAME
        for form in (g, g / f"{NAME}_cells.json", g / "cells" / "r01_c01.json",
                     root / "scans" / f"{NAME}.jpg", str(g)):
            assert resolve_game(form) == (root, NAME), form
        assert resolve_game(NAME, root) == (root, NAME)


def test_helpers():
    with tempfile.TemporaryDirectory() as t:
        root = _layout(Path(t))
        assert find_scan(root, NAME) == root / "scans" / f"{NAME}.jpg"
        assert find_scan(root, "nope") is None
        assert cells_json(root, NAME).exists()


def test_unknown_raises():
    with tempfile.TemporaryDirectory() as t:
        root = _layout(Path(t))
        for bad, dr in (("no such game", root), (Path(t), None), ("x/y", root)):
            try:
                resolve_game(bad, dr)
            except GameNotFound:
                continue
            raise AssertionError(f"{bad!r} should not resolve")


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
