#!/usr/bin/env python3
"""
review.py — Interactive review of low-confidence plate appearances.

For each unreviewed PA, shows the current reading and lets you correct it.

Usage:
  uv run python review.py
  uv run python review.py --game 2026-06-07
  uv run python review.py --game-id 1
  uv run python review.py --all          (re-review already-reviewed PAs)

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

    # Two tokens
    a, b = parts[0].upper(), parts[1].upper()
    if b in ("Y", "N"):
        return a, (1 if b == "Y" else 0), False
    if a in ("Y", "N"):
        return b, (1 if a == "Y" else 0), False
    return a, current_run, False


def _fmt_run(val: int) -> str:
    return "Yes" if val else "No "


@click.command()
@click.option("--game-id", default=None, type=int, help="Limit to a specific game id (integer)")
@click.option("--game", "game_date", default=None, help="Limit to a game date (YYYY-MM-DD)")
@click.option("--all", "show_all", is_flag=True, help="Re-review already-reviewed PAs too")
@click.option("--db", "db_path", default="data/season.db", show_default=True)
def main(game_id: int | None, game_date: str | None, show_all: bool, db_path: str) -> None:
    path = Path(db_path)
    init_db(path)
    conn = get_connection(path)

    where_clauses = ["pa.needs_review = 1"]
    params: list = []
    if not show_all:
        where_clauses.append("pa.reviewed = 0")
    if game_id is not None:
        where_clauses.append("pa.game_id = ?")
        params.append(game_id)
    if game_date is not None:
        where_clauses.append("g.date = ?")
        params.append(game_date)

    where = " AND ".join(where_clauses)
    rows = conn.execute(
        f"""SELECT pa.pa_id, p.name, g.date, g.opponent,
                   pa.inning, pa.result, pa.run_scored, pa.raw_notes,
                   pa.reviewed
             FROM plate_appearances pa
             JOIN players p ON pa.player_id = p.player_id
             JOIN games g ON pa.game_id = g.game_id
             WHERE {where}
             ORDER BY g.date ASC, pa.batting_order ASC, pa.inning ASC""",
        params,
    ).fetchall()

    if not rows:
        click.echo("No low-confidence PAs to review.")
        return

    total = len(rows)
    click.echo(f"\n{'═'*60}")
    click.echo(f"  Low-confidence PA review  —  {total} PA(s) to review")
    click.echo(f"{'═'*60}")
    click.echo("  Enter=keep  |  result (1B/K/BB/F7/6-3…)  |  y/n  |  result+y/n  |  d=delete  |  q=quit")
    click.echo(f"{'═'*60}\n")

    reviewed_count = 0
    for idx, row in enumerate(rows, 1):
        pa_id      = row["pa_id"]
        name       = row["name"]
        date       = row["date"] or "?"
        opp        = row["opponent"] or "?"
        inning     = row["inning"]
        result     = row["result"] or "?"
        run_scored = int(row["run_scored"])
        notes      = (row["raw_notes"] or "").strip()
        already    = row["reviewed"]

        tag = "  [already reviewed]" if already else ""
        click.echo(f"[{idx}/{total}]{tag}")
        click.echo(f"  {name}  •  {date} vs {opp}  •  Inning {inning}")
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
            click.echo(f"  Deleted.\n")
            reviewed_count += 1
            continue

        bb, hp, sac, sf = _derive_flags(new_result)
        conn.execute(
            """UPDATE plate_appearances
               SET result=?, run_scored=?, bb=?, hp=?, sac=?, sf=?, reviewed=1
               WHERE pa_id=?""",
            (new_result, new_run, bb, hp, sac, sf, pa_id),
        )
        conn.commit()

        changed = []
        if new_result.upper() != (result or "").upper():
            changed.append(f"result {result} → {new_result}")
        if new_run != run_scored:
            changed.append(f"run_scored {_fmt_run(run_scored)} → {_fmt_run(new_run)}")
        summary = f"  {'Updated: ' + ', '.join(changed) if changed else 'Confirmed as-is.'}\n"
        click.echo(summary)
        reviewed_count += 1

    conn.close()
    click.echo(f"Reviewed {reviewed_count}/{total} PAs this session.")


if __name__ == "__main__":
    main()
