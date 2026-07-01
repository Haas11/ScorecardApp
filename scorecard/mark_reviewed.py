"""Mark all plate appearances for a game as reviewed in the DB.

Run this after you've inspected the Low Confidence tab (and any reconciliation
warnings) and confirmed the extracted plays are correct.  Re-exports the
workbook so the Low Confidence tab no longer lists those plays.

Usage:
    uv run python mark_reviewed.py --date 2026-06-07 --opponent Almere
    uv run python mark_reviewed.py --date 2026-06-07 --opponent Almere --export --export-out ../stats.xlsx
"""
from __future__ import annotations

import click

from db import get_connection, find_duplicate_game, mark_reviewed, _DB_PATH


@click.command()
@click.option("--date", required=True, help="Game date (YYYY-MM-DD)")
@click.option("--opponent", required=True, help="Opponent name (as imported)")
@click.option("--game-number", default=None, help="Game number if doubleheader")
@click.option("--export", "do_export", is_flag=True, help="Re-export stats.xlsx after marking")
@click.option("--export-out", default="../stats.xlsx", show_default=True)
def main(date: str, opponent: str, game_number: str | None, do_export: bool, export_out: str) -> None:
    conn = get_connection(_DB_PATH)
    game_id = find_duplicate_game(conn, date, opponent, game_number)
    if game_id is None:
        print(f"Game not found in DB: {date} vs {opponent} (game_number={game_number})")
        print("Check --date and --opponent match exactly what was imported.")
        conn.close()
        raise SystemExit(1)

    n = mark_reviewed(conn, game_id)
    print(f"Marked {n} plate appearance(s) as reviewed  (game_id={game_id})")
    conn.close()

    if do_export:
        from export_season import export_season
        export_season(export_out)


if __name__ == "__main__":
    main()
