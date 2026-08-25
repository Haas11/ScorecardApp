"""
Reimport a _cells.json (or all of them) into the season DB and regenerate HTML widgets.

Usage:
  uv run python reimport.py "Quick 2026/games/2026-04-12 - Thamen (Home)/2026-04-12 - Thamen (Home)_cells.json"
  uv run python reimport.py --all "Quick 2026/games"
  uv run python reimport.py --sync-cells "Quick 2026/games/2026-04-12 - Thamen (Home)/2026-04-12 - Thamen (Home)_cells.json"

After this, run export_season.py to refresh the season xlsx.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import click

from db import get_connection, init_db, find_duplicate_game, delete_game, write_game, _DB_PATH
from models import GameExtraction
from render_widget import render_widget_for_game


def sync_cells_from_json(p: Path) -> int:
    """Write individual cell cache files from a hand-edited _cells.json.

    Reverse of the assembly step in extract_cells.py.  Use this after manually
    editing _cells.json so that --reuse-cache picks up your changes instead of
    overwriting them from the old cell cache.

    Keys only present in the cell cache (rbi_slot, confidence) are preserved.
    Returns the number of cell files written.
    """
    data = json.loads(p.read_text(encoding="utf-8"))
    cells_dir = p.parent / "cells"
    layout_path = cells_dir / "_layout.json"

    if not cells_dir.is_dir():
        raise FileNotFoundError(f"No cells/ directory next to {p.name}")
    if not layout_path.exists():
        raise FileNotFoundError(f"No _layout.json in {cells_dir}")

    col_to_inning: list[int] = json.loads(layout_path.read_text(encoding="utf-8"))["col_to_inning"]

    # inning → ordered list of column indices
    inning_to_cols: dict[int, list[int]] = {}
    for ci, inn in enumerate(col_to_inning):
        inning_to_cols.setdefault(inn, []).append(ci)

    written = 0
    for slot in data.get("lineup", []):
        ri = slot["batting_order"] - 1  # 0-indexed row

        # Flatten PAs from starter + all subs, sorted by inning
        all_pas = []
        for player in slot.get("players", []):
            for pa in player.get("plate_appearances", []):
                all_pas.append(pa)
        all_pas.sort(key=lambda pa: pa.get("inning", 0))

        inning_pa_idx: dict[int, int] = {}
        for pa in all_pas:
            inn = pa.get("inning")
            if inn is None:
                continue
            cols = inning_to_cols.get(inn, [])
            if not cols:
                continue
            idx = inning_pa_idx.get(inn, 0)
            if idx >= len(cols):
                continue
            ci = cols[idx]
            inning_pa_idx[inn] = idx + 1

            cf = cells_dir / f"r{ri+1:02d}_c{ci+1:02d}.json"

            # Preserve existing keys not present in _cells.json (e.g. rbi_slot)
            existing: dict = {}
            if cf.exists():
                try:
                    existing = json.loads(cf.read_text(encoding="utf-8"))
                except Exception:
                    pass

            existing["result"] = pa.get("result")
            existing["run"] = bool(pa.get("run_scored"))
            existing["notes"] = pa.get("notes") or None
            existing["confidence"] = pa.get("confidence", "high")
            existing["sb_count"] = int(pa.get("sb") or 0)

            cf.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
            written += 1

    return written


def reimport_one(p: Path, conn) -> str:
    """Reimport a single _cells.json. Returns a short status string."""
    data = json.loads(p.read_text(encoding="utf-8"))
    game = GameExtraction.model_validate(data)

    teams = game.game.teams
    opponent = teams.get("away") or teams.get("home") or ""
    existing = find_duplicate_game(conn, game.game.date, opponent, None)
    if existing is not None:
        delete_game(conn, existing)
    game_id = write_game(conn, game, str(p))

    game_stem = p.stem[:-len("_cells")] if p.stem.endswith("_cells") else p.stem
    widget_path = p.parent / f"{game_stem}.html"
    debug_img_path = p.parent / f"{game_stem}_grid_debug.png"
    render_widget_for_game(data, widget_path, debug_img_path=debug_img_path)

    replaced = f" (replaced id={existing})" if existing is not None else ""
    return f"game_id={game_id}{replaced}  →  {widget_path.name}"


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--all", "reimport_all", is_flag=True,
              help="Treat PATH as a games directory and reimport every _cells.json found.")
@click.option("--sync-cells", "do_sync_cells", is_flag=True,
              help="Write cell cache files from the _cells.json instead of reimporting to DB.")
def main(path: str, reimport_all: bool, do_sync_cells: bool) -> None:
    """Reimport one or all _cells.json files into the DB and regenerate HTML widgets."""

    if do_sync_cells:
        p = Path(path).resolve()
        n = sync_cells_from_json(p)
        click.echo(f"Synced {n} cell cache file(s) from {p.name}")
        click.echo("Now run crawl.py --reuse-cache to backfill without overwriting your edits.")
        return

    init_db(_DB_PATH)
    conn = get_connection(_DB_PATH)

    if reimport_all:
        games_dir = Path(path).resolve()
        cells_files = sorted(games_dir.glob("*/*_cells.json"))
        if not cells_files:
            click.echo(f"No *_cells.json files found under {games_dir}")
            conn.close()
            return
        click.echo(f"Reimporting {len(cells_files)} game(s) from {games_dir}\n")
        ok, failed = 0, 0
        for p in cells_files:
            try:
                status = reimport_one(p, conn)
                click.echo(f"  OK   {p.parent.name}  —  {status}")
                ok += 1
            except Exception as exc:
                click.echo(f"  FAIL {p.parent.name}  —  {exc}")
                failed += 1
        click.echo(f"\n{ok} imported, {failed} failed.")
    else:
        p = Path(path).resolve()
        status = reimport_one(p, conn)
        click.echo(f"DB+HTML: {status}")

    conn.close()
    click.echo("\nRun 'uv run python export_season.py' to refresh the season xlsx.")


if __name__ == "__main__":
    main()
