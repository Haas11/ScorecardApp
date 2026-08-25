"""
Crawl a games directory and re-run extract_cells.py --reuse-cache for each game.

Useful for backfilling new fields (SB, RBI, etc.) across already-analyzed games
without re-doing any VLM classification.

Usage:
  uv run python crawl.py "Quick 2026/games"
  uv run python crawl.py "Quick 2026/games" --game "2026-04-12"   # single game by date fragment
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@click.command()
@click.argument("games_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--game", "game_filter", default=None,
              help="Only process folders whose name contains this string.")
def main(games_dir: str, game_filter: str | None) -> None:
    """Re-run extract_cells --reuse-cache for every game folder under GAMES_DIR."""
    root = Path(games_dir).resolve()

    folders = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if game_filter:
        folders = [f for f in folders if game_filter in f.name]

    if not folders:
        click.echo("No matching game folders found.")
        return

    click.echo(f"Crawling {len(folders)} game(s) in {root}\n")
    ok, failed = 0, 0

    scans_dir = root.parent / "scans"

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
