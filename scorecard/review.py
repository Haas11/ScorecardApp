#!/usr/bin/env python3
"""
review.py — Interactive review and correction of plate appearances.

Default mode: shows only low-confidence PAs (needs_review=1, not yet reviewed).
--all flag:   shows EVERY PA for the game — use this to fix any wrong result
              you spotted by eye in the HTML.
--max-conf N: widen the low-confidence filter to any unreviewed PA with
              confidence <= N (1-5 scale; default filter uses the DB's
              needs_review flag, which is confidence <= config.yml's
              review.threshold).

GAME is the full game name ("2026-09-27 Herons (Home)"), any path into the
game, or a fragment of the name such as its date. Omit it to go through the
flags of every game.

A reviewed PA (corrected or confirmed) is hand-checked: confidence 5/5. After
the session the DB, the game's _cells.json, its cell cache (so --reuse-cache
keeps the fix) and the HTML widget are all updated.

Usage:
  uv run python review.py "2026-09-27 Herons (Home)" --all
  uv run python review.py 2026-04-12
  uv run python review.py --game-id 7 --all
  uv run python review.py                      (only low-confidence flags)

Input at each prompt:
  Enter          keep everything as-is, mark reviewed (hand-checked)
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

from db import find_game_ids, get_connection, init_db


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


def _patch_cells_json(json_path: Path, inning: int, batting_order: int, nth: int,
                      new_result: str, new_run: int) -> bool:
    """
    Write a hand-checked PA into _cells.json.  Returns True if it was found.
    Structure: lineup[].players[].plate_appearances[]
    Matches the nth (0-based) PA of batting_order in inning: a wrapped inning
    gives a slot two PAs in the same inning.
    """
    if not json_path.exists():
        return False
    data = json.loads(json_path.read_text(encoding="utf-8"))

    for slot in data.get("lineup", []):
        if slot.get("batting_order") != batting_order:
            continue
        pas = [pa for player in slot.get("players", [])
               for pa in player.get("plate_appearances", []) if pa.get("inning") == inning]
        if nth < len(pas):
            pas[nth].update(result=new_result, run_scored=bool(new_run), confidence=5,
                            result_conf=5, run_conf=5, conf_reasons=["hand-checked"])
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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
        debug_img_path = json_path.parent / f"{game_stem}_grid_debug.png"
        render_widget_for_game(data, widget_path, debug_img_path=debug_img_path)
        click.echo(f"  HTML regenerated: {widget_path.name}")
    except Exception as exc:
        click.echo(f"  (Could not regenerate HTML: {exc})")


@click.command()
@click.argument("game", required=False, default=None)
@click.option("--game", "game_opt", default=None, help="Same as GAME (kept for old scripts).")
@click.option("--game-id", default=None, type=int, help="Limit to a specific game id")
@click.option("--all", "show_all", is_flag=True,
              help="Show every PA for the game, not just low-confidence flags.")
@click.option("--max-conf", "max_conf", default=None, type=int,
              help="Widen the filter to any unreviewed PA with confidence <= N (1-5).")
@click.option("--db", "db_path", default=None)
def main(game: str | None, game_opt: str | None, game_id: int | None, show_all: bool,
         max_conf: int | None, db_path: str | None) -> None:
    """Walk through low-confidence plate appearances of GAME and correct them."""
    from db import _DB_PATH
    path = Path(db_path) if db_path else _DB_PATH
    init_db(path)
    conn = get_connection(path)
    game = game or game_opt

    # --all with a game filter: show every PA so you can fix any wrong result.
    # --max-conf N: any unreviewed PA with confidence <= N, wider than the
    # DB's default needs_review flag (confidence <= config.yml review.threshold).
    # Without either: only show flagged low-confidence PAs not yet reviewed.
    if show_all:
        where_clauses: list[str] = []
    elif max_conf is not None:
        where_clauses = ["pa.confidence <= ?", "pa.reviewed = 0"]
    else:
        where_clauses = ["pa.needs_review = 1", "pa.reviewed = 0"]

    params: list = [max_conf] if (not show_all and max_conf is not None) else []
    if game_id is not None:
        where_clauses.append("pa.game_id = ?")
        params.append(game_id)
    if game is not None:
        ids = find_game_ids(conn, game)
        if not ids:
            click.echo(f"No game in the DB matches {game!r}.")
            return
        where_clauses.append(f"pa.game_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)

    if show_all and game is None and game_id is None:
        click.echo("--all lists every PA of one game: give GAME or --game-id.")
        return

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    rows = conn.execute(
        f"""SELECT pa.pa_id, pa.batting_order, p.name, g.date, g.opponent,
                   pa.inning, pa.result, pa.run_scored, pa.raw_notes,
                   pa.reviewed, pa.needs_review, pa.confidence, pa.conf_reasons,
                   g.raw_json_path, pa.game_id,
                   (SELECT COUNT(*) FROM plate_appearances o
                     WHERE o.game_id = pa.game_id AND o.batting_order = pa.batting_order
                       AND o.inning = pa.inning AND o.pa_id < pa.pa_id) AS nth
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
    if show_all:
        mode = "all PAs"
    elif max_conf is not None:
        mode = f"confidence <= {max_conf}"
    else:
        mode = "low-confidence flags"
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
        confidence  = row["confidence"]
        conf_reasons = (row["conf_reasons"] or "").replace(";", "; ")
        json_path   = Path(row["raw_json_path"]) if row["raw_json_path"] else None
        gid         = row["game_id"]

        tag = ""
        if already:
            tag = "  [reviewed]"
        if confidence is not None:
            reason_part = f": {conf_reasons}" if conf_reasons else ""
            tag += f"  [conf {confidence}/5{reason_part}]"
        elif flagged:
            tag += "  [low-confidence]"  # pre-#11 row with no int confidence yet
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
               SET result=?, run_scored=?, bb=?, hp=?, sac=?, sf=?, reviewed=1, needs_review=0,
                   confidence=5, conf_reasons='hand-checked'
               WHERE pa_id=?""",
            (new_result, new_run, bb, hp, sac, sf, pa_id),
        )
        conn.commit()

        changed = []
        if new_result.upper() != (result or "").upper():
            changed.append(f"result {result} → {new_result}")
        if new_run != run_scored:
            changed.append(f"run scored {_fmt_run(run_scored)} → {_fmt_run(new_run)}")

        # Corrected or confirmed: hand-checked either way, so _cells.json and
        # the cell cache get confidence 5 too.
        if json_path and _patch_cells_json(json_path, inning, bo, row["nth"], new_result, new_run):
            changed_games.add(gid)
        click.echo(f"  Updated: {', '.join(changed)}" if changed else "  Confirmed as-is.")

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
                jp = Path(row2["raw_json_path"])
                try:
                    from reimport import sync_cells_from_json
                    sync_cells_from_json(jp)
                    click.echo(f"  Cell cache synced: {jp.parent.name}")
                except Exception as exc:  # noqa: BLE001
                    click.echo(f"  (Could not sync cell cache: {exc} — run sync-edits)")
                _regenerate_html(jp)

    click.echo(f"\nReviewed {reviewed_count}/{total} PAs this session.")


if __name__ == "__main__":
    main()
