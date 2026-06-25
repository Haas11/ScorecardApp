"""
_dump_cells.py — Dump per-cell VLM cache to CSV.

Usage:
  uv run python _dump_cells.py                  # auto-detect most recent game
  uv run python _dump_cells.py 2026-06-07_almere

Output: images/_cache/cells/{stem}/{stem}_cells.csv
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
cache_root = project_root / "images" / "_cache" / "cells"

if len(sys.argv) > 1:
    stem = sys.argv[1]
else:
    folders = sorted(
        (p for p in cache_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not folders:
        print("No cached game folders found in images/_cache/cells/", file=sys.stderr)
        sys.exit(1)
    stem = folders[0].name

cache_dir = cache_root / stem
if not cache_dir.exists():
    print(f"Cache folder not found: {cache_dir}", file=sys.stderr)
    sys.exit(1)

players_file = project_root / "players.txt"
players: list[str] = []
if players_file.exists():
    for line in players_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        players.append(parts[0])

grid: dict[tuple[int, int], dict] = {}
for f in cache_dir.glob("r??_c??.json"):
    parts = f.stem.split("_")
    ri = int(parts[0][1:])
    ci = int(parts[1][1:])
    grid[(ri, ci)] = json.loads(f.read_text(encoding="utf-8"))

rows = ["ri,ci,player,inning,result,run,confidence,notes"]
for ri in range(1, 10):
    for ci in range(1, 10):
        c = grid.get((ri, ci), {})
        result = c.get("result", "")
        run = c.get("run", False)
        confidence = c.get("confidence", "")
        notes = (c.get("notes") or "").replace(",", ";")
        player = players[ri - 1] if ri - 1 < len(players) else f"P{ri}"
        rows.append(f"{ri},{ci},{player},{ci},{result},{run},{confidence},{notes}")

out_path = cache_dir / f"{stem}_cells.csv"
out_path.write_text("\n".join(rows), encoding="utf-8")
print(f"Wrote {len(rows) - 1} rows → {out_path}")
