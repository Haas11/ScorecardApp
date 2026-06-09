from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows console may not support full Unicode — replace unencodable chars
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import click

from db import get_connection, init_db, find_duplicate_game, delete_game, write_game
from extract import extract_scorecard, extract_scorecard_two_step
from stats import compute_player_stats


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--opponent", default=None, help="Override extracted opponent name")
@click.option("--date", "date_str", default=None, help="Override extracted date (YYYY-MM-DD)")
@click.option("--dry-run", is_flag=True, help="Extract and validate but do not write to DB")
@click.option("--review", is_flag=True, help="Show PA table and prompt confirmation before writing")
@click.option("--model", default=None, help="Override EXTRACTION_MODEL from .env")
@click.option("--two-step", "two_step", is_flag=True, help="Use two-step extraction: visual scan then JSON parse")
@click.option(
    "--players", "players_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to a text file with one player name per line (roster hint for VLM)",
)
def main(
    image_path: str,
    opponent: str | None,
    date_str: str | None,
    dry_run: bool,
    review: bool,
    model: str | None,
    two_step: bool,
    players_file: str | None,
) -> None:
    db_path = Path("data/season.db")
    init_db(db_path)
    conn = get_connection(db_path)

    # Load optional roster hint (supports "Name, jersey#" format)
    player_names: list[str] | None = None
    if players_file:
        player_names = []
        for line in open(players_file, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(",")[0].strip()
            if name:
                player_names.append(name)
        print(f"Loaded {len(player_names)} player names from {players_file}")

    # Step 1-4: Extract and validate
    print(f"Extracting scorecard from {image_path}...")
    extractor = extract_scorecard_two_step if two_step else extract_scorecard
    extraction, archive_path = extractor(
        image_path,
        model=model,
        raw_dir=Path("data/raw"),
        player_names=player_names,
    )
    print(f"Raw JSON archived to {archive_path}")

    # Step 5: Apply overrides
    game = extraction.game
    date_val = date_str or game.date
    teams = game.teams
    opponent_val = opponent or teams.get("away") or teams.get("home")

    print(f"\nGame: {teams.get('away','?')} @ {teams.get('home','?')}")
    print(f"Date: {date_val}  |  Opponent: {opponent_val}  |  Game #: {game.game_number}")

    # Step 6: Duplicate check
    if not dry_run:
        existing_id = find_duplicate_game(conn, date_val, opponent_val, game.game_number)
        if existing_id is not None:
            click.echo(
                f"\nWarning: game already imported (id={existing_id}, "
                f"date={date_val}, opponent={opponent_val})."
            )
            if not click.confirm("Re-import? This will delete existing PAs for this game."):
                conn.close()
                sys.exit(0)
            delete_game(conn, existing_id)

    # Step 7: Print extraction summary
    print("\n-- Players --------------------------------------------------")
    low_conf_by_player: dict[str, list] = {}
    for slot in extraction.lineup:
        for player in slot.players:
            pa_count = len(player.plate_appearances)
            low_conf = [pa for pa in player.plate_appearances if pa.confidence == "low"]
            print(f"  [{slot.batting_order}] {player.name}: {pa_count} PA", end="")
            if low_conf:
                print(f"  ({len(low_conf)} low-confidence)", end="")
                low_conf_by_player[player.name] = low_conf
            print()

    if low_conf_by_player:
        print("\n-- Low-confidence PAs ---------------------------------------")
        for name, pas in low_conf_by_player.items():
            for pa in pas:
                print(f"  {name} inning {pa.inning}: '{pa.result}' - {pa.notes}")

    if extraction.ambiguous_cells:
        print("\n-- Ambiguous cells ------------------------------------------")
        for cell in extraction.ambiguous_cells:
            print(f"  {cell.batter} inning {cell.inning}: '{cell.raw_text}'")

    # Step 8: --review confirmation
    if review:
        print("\n-- Full PA Table --------------------------------------------")
        for slot in extraction.lineup:
            for player in slot.players:
                print(f"\n  {player.name} (order {slot.batting_order}):")
                for pa in player.plate_appearances:
                    flag = " [LOW]" if pa.confidence == "low" else ""
                    print(
                        f"    Inn {pa.inning:2d}: {pa.result:<8} "
                        f"R={int(pa.run_scored)} RBI={pa.rbi}{flag}"
                    )
        if not click.confirm("\nConfirm import?"):
            conn.close()
            sys.exit(0)

    if dry_run:
        print("\n-- Extracted JSON -------------------------------------------")
        print(json.dumps(extraction.model_dump(), indent=2, default=str))
        print("\n[dry-run] Validation passed. Nothing written to DB.")
        conn.close()
        return

    # Step 9: Write to DB
    game_id = write_game(
        conn, extraction, str(archive_path),
        opponent_override=opponent,
        date_override=date_str,
    )
    print(f"\nGame id={game_id} written to DB.")

    # Step 10: One-line stat summary per player
    print("\n-- Season stats (updated) -----------------------------------")
    for slot in extraction.lineup:
        for player in slot.players:
            # Look up player_id by name
            row = conn.execute(
                "SELECT player_id FROM players WHERE name=? OR normalized_name=?",
                (player.name, player.name.lower()),
            ).fetchone()
            if row:
                from stats import compute_player_stats
                s = compute_player_stats(conn, row["player_id"], player.name)
                avg_str = f"{s.AVG:.3f}"
                print(f"  {player.name:<20} {s.AB}-{s.H}  AVG {avg_str}")

    conn.close()


if __name__ == "__main__":
    main()
