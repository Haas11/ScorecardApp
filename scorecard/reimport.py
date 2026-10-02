"""
Reimport a _cells.json (or all of them) into the season DB and regenerate HTML widgets.

Usage (GAME = game folder, any file in it, its scan, or just the game name):
  uv run python reimport.py "2026-04-12 - Thamen (Home)"
  uv run python reimport.py "../Quick 2026/games/2026-04-12 - Thamen (Home)"
  uv run python reimport.py --all                      # every game under <data root>/games
  uv run python reimport.py --sync-cells "2026-04-12 - Thamen (Home)"

After this, run export_season.py to refresh the season xlsx.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import click

from db import (
    get_connection, init_db, find_duplicate_game, delete_game, write_game,
    get_data_root, get_db_path, DATA_ROOT_ENV_VAR,
)
from gamepaths import GameNotFound, cells_json, resolve_game
from models import GameExtraction
from render_widget import render_widget_for_game


def sync_cells_from_json(p: Path) -> int:
    """Write individual cell cache files from a hand-edited _cells.json.

    Reverse of the assembly step in extract_cells.py.  Use this after manually
    editing _cells.json so that --reuse-cache picks up your changes instead of
    overwriting them from the old cell cache.

    Keys only present in the cell cache (rbi_slot, reread, adjusted) are
    preserved; result_conf/run_conf are written back from the JSON if present.
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
            if pa.get("result_conf") is not None:
                existing["result_conf"] = pa.get("result_conf")
            if pa.get("run_conf") is not None:
                existing["run_conf"] = pa.get("run_conf")
            existing["sb_count"] = int(pa.get("sb") or 0)
            # Checked by a person (GUI Review page): full confidence from now on,
            # and any second-read verdict about the old reading is moot.
            if "hand-checked" in (pa.get("conf_reasons") or []):
                existing["hand_checked"] = True
                existing.pop("verify", None)

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
@click.argument("path", metavar="[GAME]", required=False, default=None)
@click.option("--all", "reimport_all", is_flag=True,
              help="Reimport every game: GAME is then a games directory "
                   "(default: <data root>/games).")
@click.option("--sync-cells", "do_sync_cells", is_flag=True,
              help="Write cell cache files from the _cells.json instead of reimporting to DB.")
@click.option("--data-root", "data_root_opt", default=None, envvar=DATA_ROOT_ENV_VAR,
              help="Season data root whose DB to write to (default: config.yml paths.data_root), "
                   f"e.g. \"Quick 2026 - Ex Spring Training\". Also settable via {DATA_ROOT_ENV_VAR}.")
def main(path: str, reimport_all: bool, do_sync_cells: bool, data_root_opt: str | None) -> None:
    """Reimport one or all _cells.json files into the DB and regenerate HTML widgets.

    GAME is the game folder, any file inside it (e.g. its _cells.json), its scan
    image, or just the game name (looked up under the data root)."""
    data_root = get_data_root(data_root_opt)
    if not reimport_all:
        if path is None:
            raise click.UsageError("Give a GAME, or --all to reimport every game.")
        try:
            root, name = resolve_game(path, data_root)
        except GameNotFound as exc:
            raise click.UsageError(str(exc))
        single = cells_json(root, name)
        if not single.exists():
            raise click.UsageError(f"No _cells.json for {name!r} at {single}")

    if do_sync_cells:
        if reimport_all:
            raise click.UsageError("--sync-cells works on one game at a time; drop --all.")
        p = single
        n = sync_cells_from_json(p)
        click.echo(f"Synced {n} cell cache file(s) from {p.name}")
        click.echo("Now re-run from cache (crawl.py, or: scorecard.py reread-season) "
                   "without overwriting your edits.")
        return

    db_path = get_db_path(data_root)
    init_db(db_path)
    conn = get_connection(db_path)

    if reimport_all:
        games_dir = Path(path).resolve() if path else data_root / "games"
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
        # A game whose folder was deleted leaves the DB too.
        for row in conn.execute("SELECT game_id, raw_json_path FROM games").fetchall():
            jp = Path(row["raw_json_path"] or "")
            if jp.parent.parent.resolve() == games_dir and not jp.exists():
                delete_game(conn, row["game_id"])
                click.echo(f"  REMOVED {jp.parent.name}  —  folder no longer exists")
    else:
        status = reimport_one(single, conn)
        click.echo(f"DB+HTML: {status}")

    conn.close()
    click.echo("\nRun 'uv run python export_season.py' to refresh the season xlsx.")


if __name__ == "__main__":
    main()
