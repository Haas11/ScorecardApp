"""
Reimport a (manually edited) _cells.json into the season DB and regenerate
the HTML widget.

Usage:
  uv run python reimport.py "Quick 2026 data/games/2026-04-12 - Thamen (Home)/2026-04-12 - Thamen (Home)_cells.json"

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


@click.command()
@click.argument("json_path", type=click.Path(exists=True))
def main(json_path: str) -> None:
    """Reimport an edited _cells.json into the DB and regenerate the HTML widget."""
    p = Path(json_path).resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    game = GameExtraction.model_validate(data)

    # ── DB ────────────────────────────────────────────────────────────────────
    init_db(_DB_PATH)
    conn = get_connection(_DB_PATH)
    teams = game.game.teams
    opponent = teams.get("away") or teams.get("home") or ""
    existing = find_duplicate_game(conn, game.game.date, opponent, None)
    if existing is not None:
        click.echo(f"DB    : replacing game id={existing}")
        delete_game(conn, existing)
    game_id = write_game(conn, game, str(p))
    conn.close()
    click.echo(f"DB    : imported as game id={game_id}")

    # ── HTML widget ───────────────────────────────────────────────────────────
    game_stem = p.stem[:-len("_cells")] if p.stem.endswith("_cells") else p.stem
    widget_path = p.parent / f"{game_stem}.html"
    debug_img_path = p.parent / f"{game_stem}_grid_debug.png"
    render_widget_for_game(data, widget_path, debug_img_path=debug_img_path)
    click.echo(f"HTML  : {widget_path.name}")

    click.echo("\nRun 'uv run python export_season.py' to refresh the season xlsx.")


if __name__ == "__main__":
    main()
