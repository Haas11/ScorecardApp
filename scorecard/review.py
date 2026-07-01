#!/usr/bin/env python3
"""
review.py — Interactive review and correction of plate appearances.

Default mode: shows only low-confidence PAs (needs_review=1, not yet reviewed).
--all flag:   shows EVERY PA for the game — use this to fix any wrong result
              you spotted by eye in the HTML.

After each correction the DB, the game's _cells.json, and the HTML widget are
all updated so everything stays in sync.

Usage:
  uv run python review.py --game 2026-04-12 --all
  uv run python review.py --game-id 7 --all
  uv run python review.py                      (only low-confidence flags)

Input at each prompt:
  Enter          keep everything as-is, mark reviewed
  1B / K / BB    change the result  (keeps run_scored)
  y / n          change run_scored only  (y=scored, n=did not score)
  1B y           change result AND set run_scored=yes
  FC n           change result AND set run_scored=no
  d              delete this PA entirely
  q              quit (all changes made so far are saved)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from db import get_connection, init_db


def _derive_flags(result: str) -> tuple[int, int, int, int]:
    """Return (bb, hp, sac, sf) from a result string."""
    r = result.upper()
    return (
        1 if r == "BB" else 0,
        1 if r in ("HBP", "HP") else 0,
        1 if r in ("SAC", "SH") else 0,
        1 if r == "SF" else 0,
    )


def _parse_input(raw: str, current_result: str, current_run: int):
    """
    Parse user input.
    Returns (new_result, new_run_scored, delete) or None to signal quit.
    """
    parts = raw.strip().split()
    if not parts:
        return current_result, current_run, False

    if len(parts) == 1:
        token = parts[0].upper()
        if token == "Q":
            return None
        if token == "D":
            return current_result, current_run, True
        if token == "Y":
            return current_result, 1, False
        if token == "N":
            return current_result, 0, False
        return token, current_run, False

    a, b = parts[0].upper(), parts[1].upper()
    if b in ("Y", "N"):
        return a, (1 if b == "Y" else 0), False
    if a in ("Y", "N"):
        return b, (1 if a == "Y" else 0), False
    return a, current_run, False


def _fmt_run(val: int) -> str:
    return "Yes" if val else "No "


def _patch_cells_json(json_path: Path, player_name: str, inning: int,
                      batting_order: int, new_result: str, new_run: int) -> bool:
    """
    Update the matching PA inside _cells.json.  Returns True if a change was made.
    Structure: lineup[].players[].plate_appearances[]
    Matches by batting_order + inning.
    """
    if not json_path.exists():
        return False
    data = json.loads(json_path.read_text(encoding="utf-8"))

    for slot in data.get("lineup", []):
        if slot.get("batting_order") != batting_order:
            continue
        for player in slot.get("players", []):
            for pa in player.get("plate_appearances", []):
                if pa.get("inning") == inning:
                    pa["result"] = new_result
                    pa["run_scored"] = bool(new_run)
                    json_path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    return True
    return False


def _regenerate_html(json_path: Path) -> None:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from render_widget import render_widget_for_game
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # stem is e.g. "2026-04-12 - Thamen (Home)_cells" → strip "_cells"
        game_stem = json_path.stem[:-len("_cells")] if json_path.stem.endswith("_cells") else json_path.stem
        widget_path = json_path.parent / f"{game_stem}.html"
        render_widget_for_game(data, widget_path)
        click.echo(f"  HTML regenerated: {widget_path.name}")
    except Exception as exc:
        click.echo(f"  (Could not regenerate HTML: {exc})")


@click.command()
@click.option("--game-id", default=None, type=int, help="Limit to a specific game id")
@click.option("--game", "game_date", default=None, help="Limit to a game date (YYYY-MM-DD)")
@click.option("--all", "show_all", is_flag=True,
              help="Show every PA for the game, not just low-confidence flags.")
@click.option("--db", "db_path", default=None)
def main(game_id: int | None, game_date: str | None, show_all: bool,
         db_path: str | None) -> None:
    from db import _DB_PATH
    path = Path(db_path) if db_path else _DB_PATH
    init_db(path)
    conn = get_connection(path)

    # --all with a game filter: show every PA so you can fix any wrong result.
    # Without --all: only show flagged low-confidence PAs not yet reviewed.
    if show_all:
        where_clauses: list[str] = []
    else:
        where_clauses = ["pa.needs_review = 1", "pa.reviewed = 0"]

    params: list = []
    if game_id is not None:
        where_clauses.append("pa.game_id = ?")
        params.append(game_id)
    if game_date is not None:
        where_clauses.append("g.date LIKE ?")
        params.append(f"{game_date}%")

    if not where_clauses and not show_all:
        click.echo("Specify --game or --game-id to review a specific game.")
        return

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    rows = conn.execute(
        f"""SELECT pa.pa_id, pa.batting_order, p.name, g.date, g.opponent,
                   pa.inning, pa.result, pa.run_scored, pa.raw_notes,
                   pa.reviewed, pa.needs_review, g.raw_json_path, pa.game_id
             FROM plate_appearances pa
             JOIN players p ON pa.player_id = p.player_id
             JOIN games g ON pa.game_id = g.game_id
             {where}
             ORDER BY g.date ASC, pa.batting_order ASC, pa.inning ASC""",
        params,
    ).fetchall()

    if not rows:
        click.echo("No PAs to review.")
        return

    total = len(rows)
    mode = "all PAs" if show_all else "low-confidence flags"
    click.echo(f"\n{'═'*60}")
    click.echo(f"  Review ({mode})  —  {total} PA(s)")
    click.echo(f"{'═'*60}")
    click.echo("  Enter=keep  |  result (1B/K/BB/F7…)  |  y/n (run)  |  result+y/n  |  d=delete  |  q=quit")
    click.echo(f"{'═'*60}\n")

    changed_games: set[int] = set()
    reviewed_count = 0

    for idx, row in enumerate(rows, 1):
        pa_id       = row["pa_id"]
        bo          = row["batting_order"]
        name        = row["name"]
        date        = row["date"] or "?"
        opp         = row["opponent"] or "?"
        inning      = row["inning"]
        result      = row["result"] or "?"
        run_scored  = int(row["run_scored"])
        notes       = (row["raw_notes"] or "").strip()
        already     = row["reviewed"]
        flagged     = row["needs_review"]
        json_path   = Path(row["raw_json_path"]) if row["raw_json_path"] else None
        gid         = row["game_id"]

        tag = ""
        if already:
            tag = "  [reviewed]"
        if flagged:
            tag += "  [low-confidence]"
        click.echo(f"[{idx}/{total}]{tag}")
        click.echo(f"  #{bo} {name}  •  {date} vs {opp}  •  Inning {inning}")
        click.echo(f"  Result: {result:<8}  Run scored: {_fmt_run(run_scored)}")
        if notes:
            click.echo(f"  Notes:  {notes[:120]}")
        click.echo()

        raw = click.prompt("  > ", default="", show_default=False, prompt_suffix="").strip()
        parsed = _parse_input(raw, result, run_scored)

        if parsed is None:
            click.echo("\nQuitting — all changes so far have been saved.")
            break

        new_result, new_run, delete = parsed

        if delete:
            conn.execute("DELETE FROM plate_appearances WHERE pa_id = ?", (pa_id,))
            conn.commit()
            click.echo("  Deleted.\n")
            reviewed_count += 1
            changed_games.add(gid)
            continue

        bb, hp, sac, sf = _derive_flags(new_result)
        conn.execute(
            """UPDATE plate_appearances
               SET result=?, run_scored=?, bb=?, hp=?, sac=?, sf=?, reviewed=1, needs_review=0
               WHERE pa_id=?""",
            (new_result, new_run, bb, hp, sac, sf, pa_id),
        )
        conn.commit()

        changed = []
        if new_result.upper() != (result or "").upper():
            changed.append(f"result {result} → {new_result}")
        if new_run != run_scored:
            changed.append(f"run scored {_fmt_run(run_scored)} → {_fmt_run(new_run)}")

        if changed:
            click.echo(f"  Updated: {', '.join(changed)}")
            # Keep _cells.json in sync
            if json_path:
                patched = _patch_cells_json(json_path, name, inning, bo, new_result, new_run)
                if patched:
                    click.echo(f"  Patched {json_path.name}")
            changed_games.add(gid)
        else:
            click.echo("  Confirmed as-is.")

        click.echo()
        reviewed_count += 1

    conn.close()

    # Regenerate HTML for any game that had corrections
    if changed_games:
        for gid in changed_games:
            conn2 = get_connection(path)
            row2 = conn2.execute(
                "SELECT raw_json_path FROM games WHERE game_id=?", (gid,)
            ).fetchone()
            conn2.close()
            if row2 and row2["raw_json_path"]:
                _regenerate_html(Path(row2["raw_json_path"]))

    click.echo(f"\nReviewed {reviewed_count}/{total} PAs this session.")


if __name__ == "__main__":
    main()
