"""
Resolve what the user pointed a command at into a game (#20).

Every command that works on one game accepts any of:
  - the game folder                      ../Quick 2026/games/2026-07-12 Grizzlies (Home)
  - any file inside it                   .../2026-07-12 Grizzlies (Home)_cells.json, .html, cells/r01_c01.json
  - the scan image                       ../Quick 2026/scans/2026-07-12 Grizzlies (Home).jpg
  - just the game name                   "2026-07-12 Grizzlies (Home)"  (looked up under <data root>/games/)

Layout (fixed by the pipeline): <data root>/scans/<name>.<ext> and
<data root>/games/<name>/<name>_cells.json.
"""
from __future__ import annotations

from pathlib import Path

SCAN_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


class GameNotFound(ValueError):
    pass


def resolve_game(path: str | Path, data_root: Path | None = None) -> tuple[Path, str]:
    """Return (data_root, game_name) for any of the accepted forms."""
    p = Path(path)
    if p.exists():
        p = p.resolve()
        if p.is_file() and p.suffix in SCAN_EXTS and p.parent.name == "scans":
            return p.parent.parent, p.stem
        # Walk up to the folder directly under ".../games/".
        for d in [p] + list(p.parents):
            if d.parent.name == "games" and d.is_dir():
                return d.parent.parent, d.name
        raise GameNotFound(f"{path} is not a scan in scans/ or inside a games/<game> folder")
    if data_root is not None and len(p.parts) == 1:
        if (data_root / "games" / p.name).is_dir() or find_scan(data_root, p.name):
            return data_root, p.name
    raise GameNotFound(f"Could not find a game for {path!r}"
                       + (f" (also looked under {data_root / 'games'})" if data_root else ""))


def game_folder(data_root: Path, name: str) -> Path:
    return data_root / "games" / name


def cells_json(data_root: Path, name: str) -> Path:
    return game_folder(data_root, name) / f"{name}_cells.json"


def find_scan(data_root: Path, name: str) -> Path | None:
    for ext in SCAN_EXTS:
        cand = data_root / "scans" / f"{name}{ext}"
        if cand.exists():
            return cand
    return None
