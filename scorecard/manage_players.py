from __future__ import annotations

from pathlib import Path

import click

from db import get_connection, init_db, normalize_name

_DB_PATH = Path("data/season.db")


@click.group()
def cli() -> None:
    pass


@cli.command("aliases")
def review_aliases() -> None:
    """Review pending fuzzy matches from last import."""
    init_db(_DB_PATH)
    conn = get_connection(_DB_PATH)

    rows = conn.execute(
        "SELECT id, incoming, stored, player_id, score, match_type FROM pending_aliases"
    ).fetchall()

    if not rows:
        print("No pending alias reviews.")
        conn.close()
        return

    for row in rows:
        answer = input(
            f"\nAre '{row['incoming']}' and '{row['stored']}' the same player? [y/n]: "
        ).strip().lower()
        if answer == "y":
            norm = normalize_name(row["incoming"])
            conn.execute(
                "INSERT INTO player_aliases (alias, normalized_alias, player_id) VALUES (?,?,?)",
                (row["incoming"], norm, row["player_id"]),
            )
        else:
            # Create new player and reassign PAs attributed to old player via this alias match
            new_id = conn.execute(
                "INSERT INTO players (name, normalized_name) VALUES (?,?)",
                (row["incoming"], normalize_name(row["incoming"])),
            ).lastrowid
            # Only reassign PAs where the name matched via this pending alias (no safe way
            # to do it retroactively without a name column on PAs; record alias instead)
            print(f"  Created new player '{row['incoming']}' (id={new_id}).")
            print("  Note: existing PAs were imported under the matched player.")
            print("  Use --merge to reassign if needed.")
        conn.commit()

    conn.execute("DELETE FROM pending_aliases")
    conn.commit()
    print("\nAlias review complete.")
    conn.close()


@cli.command("list")
def list_players() -> None:
    """List all players and their aliases."""
    init_db(_DB_PATH)
    conn = get_connection(_DB_PATH)

    players = conn.execute(
        "SELECT player_id, name, jersey_number FROM players ORDER BY name"
    ).fetchall()

    for p in players:
        aliases = conn.execute(
            "SELECT alias FROM player_aliases WHERE player_id=?", (p["player_id"],)
        ).fetchall()
        alias_str = ", ".join(a["alias"] for a in aliases) if aliases else "—"
        jersey = f" #{p['jersey_number']}" if p["jersey_number"] else ""
        print(f"  [{p['player_id']}] {p['name']}{jersey}  aliases: {alias_str}")

    conn.close()


@cli.command("merge")
@click.argument("name1")
@click.argument("name2")
def merge_players(name1: str, name2: str) -> None:
    """Merge NAME2 into NAME1, moving all PAs and adding alias."""
    init_db(_DB_PATH)
    conn = get_connection(_DB_PATH)

    def find_player(name: str) -> int | None:
        row = conn.execute(
            "SELECT player_id FROM players WHERE name=? OR normalized_name=?",
            (name, normalize_name(name)),
        ).fetchone()
        return row["player_id"] if row else None

    id1 = find_player(name1)
    id2 = find_player(name2)

    if id1 is None:
        print(f"Player '{name1}' not found.")
        conn.close()
        return
    if id2 is None:
        print(f"Player '{name2}' not found.")
        conn.close()
        return
    if id1 == id2:
        print("Same player — nothing to merge.")
        conn.close()
        return

    conn.execute(
        "UPDATE plate_appearances SET player_id=? WHERE player_id=?", (id1, id2)
    )
    conn.execute(
        "UPDATE pitching_appearances SET player_id=? WHERE player_id=?", (id1, id2)
    )
    conn.execute(
        "UPDATE player_aliases SET player_id=? WHERE player_id=?", (id1, id2)
    )
    norm2 = conn.execute(
        "SELECT normalized_name, name FROM players WHERE player_id=?", (id2,)
    ).fetchone()
    conn.execute(
        "INSERT INTO player_aliases (alias, normalized_alias, player_id) VALUES (?,?,?)",
        (norm2["name"], norm2["normalized_name"], id1),
    )
    conn.execute("DELETE FROM players WHERE player_id=?", (id2,))
    conn.commit()
    print(f"Merged '{name2}' into '{name1}'. '{name2}' added as alias.")
    conn.close()


if __name__ == "__main__":
    cli()
