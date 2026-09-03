"""
Copy game HTML files and the season stats xlsx to a destination folder.

Searches the source folder and one level of subfolders for *.html files,
then copies them alongside the xlsx stats file.

Usage:
  uv run python publish.py "Quick 2026 data" "C:/Shares/Quick/stats"
  uv run python publish.py "Quick 2026 data" "C:/Shares/Quick/stats" --xlsx "Quick 2026/Quick 2026 stats.xlsx"
  uv run python publish.py --data-root "Quick 2026 - Ex Spring Training" "C:/Shares/Quick/stats"
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from db import get_data_root, DATA_ROOT_ENV_VAR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def publish(source: Path, dest: Path, xlsx: Path | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    # Collect HTML files: source root + up to two subfolder levels
    html_files: list[Path] = (
        list(source.glob("*.html"))
        + list(source.glob("*/*.html"))
        + list(source.glob("*/*/*.html"))
    )

    if not html_files:
        click.echo("No HTML files found.")
    for src in sorted(html_files):
        target = dest / src.name
        shutil.copy2(src, target)
        click.echo(f"  copied  {src.relative_to(source.parent)}  →  {target.name}")

    # Copy xlsx
    if xlsx:
        if not xlsx.exists():
            click.echo(f"  WARNING: xlsx not found at {xlsx}", err=True)
        else:
            target_xlsx = dest / xlsx.name
            shutil.copy2(xlsx, target_xlsx)
            click.echo(f"  copied  {xlsx.name}  →  {target_xlsx.name}")

    click.echo(f"\nDone — {len(html_files)} HTML file(s) published to {dest}")


@click.command()
@click.argument("path_args", metavar="[SOURCE] DEST", nargs=-1, required=True)
@click.option(
    "--xlsx",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to the season stats xlsx (auto-detected if omitted).",
)
@click.option("--data-root", "data_root_opt", default=None, envvar=DATA_ROOT_ENV_VAR,
              help="Season data root to use as SOURCE (SOURCE may then be omitted), "
                   f"e.g. \"Quick 2026 - Ex Spring Training\". Also settable via {DATA_ROOT_ENV_VAR}.")
def main(path_args: tuple[str, ...], xlsx: str | None, data_root_opt: str | None) -> None:
    """Copy game HTML files (and the season xlsx) from SOURCE to DEST.

    SOURCE may be omitted when --data-root (or SCORECARD_DATA_ROOT) is set —
    it is then used as SOURCE, and PATH_ARGS is just DEST.
    """
    if len(path_args) == 2:
        source, dest = path_args
    elif len(path_args) == 1 and data_root_opt:
        source, dest = str(get_data_root(data_root_opt)), path_args[0]
    else:
        raise click.UsageError("Usage: publish.py [SOURCE] DEST [--data-root ROOT] "
                                "(SOURCE required unless --data-root is given).")
    src_path = Path(source).resolve()
    if not src_path.is_dir():
        raise click.UsageError(f"Source directory not found: {src_path}")

    xlsx_path: Path | None
    if xlsx:
        xlsx_path = Path(xlsx).resolve()
    else:
        # Auto-detect: look for *.xlsx one level up from source or inside source
        candidates = list(src_path.parent.glob("*.xlsx")) + list(src_path.glob("*.xlsx"))
        # Ignore Excel lock files (~$...)
        candidates = [p for p in candidates if not p.name.startswith("~$")]
        xlsx_path = candidates[0] if candidates else None
        if xlsx_path:
            click.echo(f"Auto-detected xlsx: {xlsx_path.name}")
        else:
            click.echo("No xlsx found — skipping stats file.")

    publish(src_path, Path(dest).resolve(), xlsx_path)


if __name__ == "__main__":
    main()
