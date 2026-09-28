"""
One entry point for every scorecard command, with plain-language names (#21).

    uv run python scorecard.py --help                 # list all commands
    uv run python scorecard.py read-scorecard GAME    # same as extract_cells.py GAME
    uv run python scorecard.py <command> --help       # options of one command

GAME can be the scan, the game folder, any file in it, or just the game name.
The old script names (extract_cells.py, reimport.py, ...) keep working; this
file only maps friendlier names onto them. Each command's module is imported
only when that command runs, so --help and small commands stay fast.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import click

HERE = Path(__file__).parent

# name -> (module, attribute, one-line help). Order = order shown in --help.
_COMMANDS: dict[str, tuple[str, str, str]] = {
    "read-scorecard": ("extract_cells", "main",
                       "Read a scanned scorecard into stats: grid, cell reading, checks, DB, widget."),
    "reread-season": ("crawl", "main",
                      "Re-run read-scorecard from cache for every game (e.g. after a pipeline change)."),
    "review-cells": ("review", "main",
                     "Walk through low-confidence plate appearances and correct them."),
    "mark-reviewed": ("mark_reviewed", "main",
                      "Mark all plate appearances of a game as reviewed."),
    "reimport-game": ("reimport", "main",
                      "Load a (hand-edited) game _cells.json into the DB and redraw its widget."),
    "fix-date": ("correct_date", "main",
                 "Correct a game's date: renames folder/files and patches JSON + DB."),
    "players": ("manage_players", "cli",
                "Player names: list, confirm aliases, merge duplicates."),
    "export-excel": ("export_season", "main",
                     "Write the season Excel workbook from the DB."),
    "publish": ("publish", "main",
                "Copy the HTML widgets and the Excel workbook to a destination folder."),
    "test-accuracy": ("eval_cells", "main",
                      "Measure cell-reading accuracy on the hand-verified games (uses API calls)."),
}


class _LazyGroup(click.Group):
    def list_commands(self, ctx):
        return list(_COMMANDS) + [c for c in super().list_commands(ctx) if c not in _COMMANDS]

    def get_command(self, ctx, name):
        if name in _COMMANDS:
            module, attr, _ = _COMMANDS[name]
            cmd = getattr(importlib.import_module(module), attr)
            cmd.help = cmd.help or _COMMANDS[name][2]
            return cmd
        return super().get_command(ctx, name)

    def format_commands(self, ctx, formatter):
        # Use the short help from the table so --help doesn't import every module.
        rows = [(n, h) for n, (_, _, h) in _COMMANDS.items()]
        rows += [(n, self.commands[n].get_short_help_str(80)) for n in sorted(self.commands)]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


@click.group(cls=_LazyGroup, context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Scorecard pipeline. GAME = scan, game folder, any file in it, or the game name."""


@cli.command("sync-edits")
@click.argument("game")
@click.option("--data-root", "data_root_opt", default=None)
def sync_edits(game, data_root_opt):
    """Copy hand edits in a game's _cells.json back into its cell cache (do this before reread-season)."""
    from reimport import main as reimport_main
    args = ["--sync-cells", game] + (["--data-root", data_root_opt] if data_root_opt else [])
    reimport_main.main(args=args, prog_name="scorecard.py sync-edits")


@cli.command("draw-widget")
@click.argument("game")
@click.option("--no-open", is_flag=True, help="Don't open the widget in the browser.")
def draw_widget(game, no_open):
    """Redraw a game's HTML widget from its _cells.json."""
    import render_widget
    sys.argv = ["render_widget.py", game] + (["--no-open"] if no_open else [])
    render_widget.main()


@cli.command("run-tests")
def run_tests():
    """Run all offline tests (no API calls) and report pass/fail."""
    tests = sorted(HERE.glob("test_*.py"))
    tests = [t for t in tests if t.name != "test_gemini_key.py"]  # needs the API
    failed = []
    for t in tests:
        r = subprocess.run([sys.executable, str(t)], cwd=HERE, capture_output=True, text=True)
        ok = r.returncode == 0
        click.echo(f"{'PASS' if ok else 'FAIL'}  {t.name}")
        if not ok:
            failed.append(t.name)
            click.echo((r.stdout + r.stderr)[-1500:])
    r = subprocess.run([sys.executable, "eval_cells.py", "--check"], cwd=HERE, capture_output=True, text=True)
    click.echo(f"{'PASS' if r.returncode == 0 else 'FAIL'}  eval_cells.py --check")
    if r.returncode != 0:
        failed.append("eval_cells --check")
    click.echo(f"\n{len(failed)} failed" if failed else "\nall passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    cli()
