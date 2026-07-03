"""
Reimport a _cells.json (or all of them) into the season DB and regenerate HTML widgets.

Usage:
  uv run python reimport.py "Quick 2026/games/2026-04-12 - Thamen (Home)/2026-04-12 - Thamen (Home)_cells.json"
  uv run python reimport.py --all "Quick 2026/games"

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
def main(path: str, reimport_all: bool) -> None:
    """Reimport one or all _cells.json files into the DB and regenerate HTML widgets."""
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
