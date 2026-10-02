"""Mark all plate appearances for a game as reviewed in the DB.

Run this after you've inspected the Low Confidence tab (and any reconciliation
warnings) and confirmed the extracted plays are correct.  Re-exports the
workbook so the Low Confidence tab no longer lists those plays.

GAME is the full game name, any path into the game, or a fragment such as its date.

Usage:
    uv run python mark_reviewed.py "2026-06-07 - Almere (Away)"
    uv run python mark_reviewed.py 2026-06-07 --export --export-out ../stats.xlsx
    uv run python mark_reviewed.py --date 2026-06-07 --opponent Almere     (old form)
"""
from __future__ import annotations

import click

from db import get_connection, find_duplicate_game, find_game_ids, mark_reviewed, _DB_PATH


@click.command()
@click.argument("game", required=False, default=None)
@click.option("--date", default=None, help="Game date (YYYY-MM-DD); old form, use GAME instead")
@click.option("--opponent", default=None, help="Opponent name (as imported); with --date")
@click.option("--game-number", default=None, help="Game number if doubleheader")
@click.option("--export", "do_export", is_flag=True, help="Re-export stats.xlsx after marking")
@click.option("--export-out", default="../stats.xlsx", show_default=True)
def main(game: str | None, date: str | None, opponent: str | None, game_number: str | None,
         do_export: bool, export_out: str) -> None:
    """Mark all plate appearances of GAME as reviewed."""
    conn = get_connection(_DB_PATH)
    if game is not None:
        ids = find_game_ids(conn, game)
        if len(ids) != 1:
            conn.close()
            raise click.UsageError(f"{game!r} matches {len(ids)} games in the DB — be more specific.")
        game_id = ids[0]
    elif date and opponent:
        game_id = find_duplicate_game(conn, date, opponent, game_number)
        if game_id is None:
            conn.close()
            raise click.UsageError(f"Game not found in DB: {date} vs {opponent} (game_number={game_number})")
    else:
        conn.close()
        raise click.UsageError("Give GAME (e.g. \"2026-06-07 - Almere (Away)\") or --date and --opponent.")

    n = mark_reviewed(conn, game_id)
    print(f"Marked {n} plate appearance(s) as reviewed  (game_id={game_id})")
    conn.close()

    if do_export:
        from export_season import export_season
        export_season(export_out)


if __name__ == "__main__":
    main()
