"""
Fix a mistyped game date (e.g. month/day swapped) without re-running the VLM
extraction. Renames the game folder, _cells.json, and .html (the per-cell
cache in cells/ moves automatically since it lives inside the renamed
folder), patches the date inside _cells.json and the DB, and regenerates the
HTML widget.

Usage:
  uv run python rename_game.py "../Quick 2026/games/2026-04-12 - Thamen (Home)" 2026-12-04
  uv run python rename_game.py "../Quick 2026/games/2026-04-12 - Thamen (Home)/2026-04-12 - Thamen (Home)_cells.json" 2026-12-04
  uv run python rename_game.py --data-root "Quick 2026 - Ex Spring Training" "2026-04-12 - Thamen (Home)" 2026-12-04
  uv run python rename_game.py --dry-run "2026-04-12 - Thamen (Home)" 2026-12-04
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

from db import get_connection, get_data_root, get_db_path, DATA_ROOT_ENV_VAR
from render_widget import render_widget_for_game

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STEM_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(.*)$")


def _resolve_game_folder(game_path: Path, data_root_opt: str | None) -> Path:
    """GAME may be a game folder, a _cells.json path, or a bare folder name
    (resolved under <data_root>/games/)."""
    if game_path.is_dir():
        return game_path
    if game_path.is_file() and game_path.name.endswith("_cells.json"):
        return game_path.parent
    if not game_path.is_absolute() and not game_path.exists():
        candidate = get_data_root(data_root_opt) / "games" / game_path
        if candidate.is_dir():
            return candidate
    raise click.UsageError(f"Could not resolve a game folder from {game_path}")


@click.command()
@click.argument("game", type=click.Path())
@click.argument("new_date")
@click.option("--data-root", "data_root_opt", default=None, envvar=DATA_ROOT_ENV_VAR,
              help="Season data root (default: config.yml paths.data_root). "
                   f"Also settable via {DATA_ROOT_ENV_VAR}.")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would change, do nothing.")
def main(game: str, new_date: str, data_root_opt: str | None, dry_run: bool) -> None:
    """Rename a game (folder + files) to correct its date to NEW_DATE (YYYY-MM-DD)."""
    if not _DATE_RE.match(new_date):
        raise click.UsageError(f"NEW_DATE must be YYYY-MM-DD, got {new_date!r}")

    old_folder = _resolve_game_folder(Path(game), data_root_opt)
    old_stem = old_folder.name
    m = _STEM_DATE_RE.match(old_stem)
    if not m:
        raise click.UsageError(f"Game folder name doesn't start with a date: {old_stem!r}")
    old_date, suffix = m.group(1), m.group(2)
    new_stem = new_date + suffix

    if old_date == new_date:
        click.echo("New date is the same as the old date — nothing to do.")
        return

    cells_json = old_folder / f"{old_stem}_cells.json"
    if not cells_json.exists():
        raise click.UsageError(f"No _cells.json found at {cells_json}")
    data = json.loads(cells_json.read_text(encoding="utf-8"))
    teams = (data.get("game") or {}).get("teams") or {}
    opponent = teams.get("away") or teams.get("home")
    game_number = (data.get("game") or {}).get("game_number")

    new_folder = old_folder.parent / new_stem
    if new_folder.exists():
        raise click.UsageError(f"Target folder already exists: {new_folder}")

    scans_dir = old_folder.parent.parent / "scans"
    old_scan = next((scans_dir / f"{old_stem}.{ext}" for ext in ("jpg", "jpeg", "png")
                      if (scans_dir / f"{old_stem}.{ext}").exists()), None)

    click.echo(f"Rename : {old_stem}  →  {new_stem}")
    click.echo(f"Folder : {old_folder}  →  {new_folder}")
    if old_scan:
        click.echo(f"Scan   : {old_scan.name}  →  {new_stem}{old_scan.suffix}")
    if dry_run:
        click.echo("(dry run — no changes made)")
        return

    old_folder.rename(new_folder)
    if old_scan:
        old_scan.rename(old_scan.with_name(f"{new_stem}{old_scan.suffix}"))

    # Rename every stem-prefixed file inside the folder (cells_cells.json, .html,
    # _run.log, _grid_debug.png, ...). The cells/ subfolder isn't stem-prefixed
    # so it moves untouched with the folder rename above.
    for f in list(new_folder.iterdir()):
        if f.is_file() and f.name.startswith(old_stem):
            f.rename(f.with_name(new_stem + f.name[len(old_stem):]))

    new_cells_json = new_folder / f"{new_stem}_cells.json"
    data.setdefault("game", {})["date"] = new_date
    new_cells_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"Patched: {new_cells_json.name}  (game.date = {new_date})")

    db_path = get_db_path(get_data_root(data_root_opt))
    if db_path.exists():
        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT game_id FROM games WHERE date=? AND opponent=? AND "
            + ("game_number IS NULL" if game_number is None else "game_number=?"),
            (old_date, opponent) if game_number is None else (old_date, opponent, game_number),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE games SET date=?, raw_json_path=? WHERE game_id=?",
                (new_date, str(new_cells_json), row["game_id"]),
            )
            conn.commit()
            click.echo(f"Patched: DB games row game_id={row['game_id']} (date + raw_json_path)")
        else:
            click.echo("DB     : no matching game row found — nothing to patch there.")
        conn.close()
    else:
        click.echo(f"DB     : {db_path.name} doesn't exist yet — nothing to patch.")

    debug_img_path = new_folder / f"{new_stem}_grid_debug.png"
    widget_path = new_folder / f"{new_stem}.html"
    render_widget_for_game(data, widget_path, debug_img_path=debug_img_path)
    click.echo(f"Widget : regenerated {widget_path.name}")

    click.echo(f"\nDone — {old_stem} is now {new_stem}.")
    click.echo("Run 'uv run python export_season.py' to refresh the season xlsx (game dates changed).")


if __name__ == "__main__":
    main()
