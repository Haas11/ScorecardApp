"""
Crawl a games directory and re-run extract_cells.py --reuse-cache for each game.

Useful for backfilling new fields (SB, RBI, etc.) across already-analyzed games
without re-doing any VLM classification.

Usage:
  uv run python crawl.py "Quick 2026/games"
  uv run python crawl.py "Quick 2026/games" --game "2026-04-12"   # single game by date fragment
  uv run python crawl.py "Quick 2026/games/2026-04-12 - Thamen (Home)"   # a single game folder also works
  uv run python crawl.py --data-root "Quick 2026 - Ex Spring Training"
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

from db import get_data_root, DATA_ROOT_ENV_VAR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@click.command()
@click.argument("games_dir", type=click.Path(file_okay=False), required=False, default=None)
@click.option("--game", "game_filter", default=None,
              help="Only process folders whose name contains this string.")
@click.option("--data-root", "data_root_opt", default=None, envvar=DATA_ROOT_ENV_VAR,
              help="Season data root; <data_root>/games is used when GAMES_DIR is omitted (default: config.yml), "
                   f"e.g. \"Quick 2026 - Ex Spring Training\". Also settable via {DATA_ROOT_ENV_VAR}.")
def main(games_dir: str | None, game_filter: str | None, data_root_opt: str | None) -> None:
    """Re-run extract_cells --reuse-cache for every game folder under GAMES_DIR."""
    if games_dir is None:
        # --data-root > env var > config.yml, same as every other command.
        games_dir = str(get_data_root(data_root_opt) / "games")
    root = Path(games_dir).resolve()
    if not root.is_dir():
        raise click.UsageError(f"Games directory not found: {root}")
    # Propagate to the extract_cells.py subprocess so its DB write targets the
    # same season (db.py reads this env var when no explicit override is passed).
    if data_root_opt:
        os.environ[DATA_ROOT_ENV_VAR] = str(get_data_root(data_root_opt))

    # Accept a single game folder too (it has a cells/ subfolder of its own):
    # treat it as the one game to crawl instead of iterating its children,
    # which would only find "cells" and skip it.
    if (root / "cells").is_dir():
        folders = [root]
        games_root = root.parent
    else:
        folders = sorted(
            d for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        games_root = root
    if game_filter:
        folders = [f for f in folders if game_filter in f.name]

    if not folders:
        click.echo("No matching game folders found.")
        return

    click.echo(f"Crawling {len(folders)} game(s) in {games_root}\n")
    ok, failed = 0, 0

    scans_dir = games_root.parent / "scans"

    for folder in folders:
        cells_dir = folder / "cells"
        if not cells_dir.is_dir():
            click.echo(f"  SKIP {folder.name}  — no cells/ folder")
            continue

        # extract_cells.py derives game_dir from the image path, so we need it
        # even with --reuse-cache. Scan lives in ../scans/<folder-name>.<ext>.
        image = None
        for ext in ("jpg", "jpeg", "png"):
            candidate = scans_dir / f"{folder.name}.{ext}"
            if candidate.exists():
                image = candidate
                break
        if image is None:
            click.echo(f"  SKIP {folder.name}  — no scan image in {scans_dir.name}/")
            continue

        click.echo(f"  → {folder.name}")

        cmd = [
            sys.executable, "extract_cells.py",
            str(image),
            "--reuse-cache", "--yes",
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent)

        if result.returncode == 0:
            ok += 1
        else:
            click.echo(f"  FAIL {folder.name}  — exit code {result.returncode}")
            failed += 1

    click.echo(f"\nDone — {ok} succeeded, {failed} failed.")
    if ok > 0:
        click.echo("Run 'uv run python reimport.py --all <games_dir>' then 'uv run python export_season.py' to refresh DB and xlsx.")


if __name__ == "__main__":
    main()
