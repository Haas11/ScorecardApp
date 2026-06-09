# Follow-up Prompts
# Use these in order within the same Claude Code session after Prompt 1 completes.
# Only move to the next prompt when the previous one is verified working.

---

## Prompt 2 — Smoke test with real image

Use after: project scaffolded, .env configured with API key

---

I have a real scorecard image to test with. Do the following:

1. Run `python process_game.py <image_path> --dry-run` against the image
2. Pretty-print the extracted JSON to console so I can review it
3. List any validation warnings or low-confidence cells
4. Do NOT write to the database yet

If extraction fails, show the raw API response and error so we can debug
the system prompt or schema.

---

## Prompt 3 — Fix extraction issues (use if Prompt 2 reveals problems)

Use after: dry-run reveals specific extraction errors

---

The dry-run extraction has the following issues:

[DESCRIBE WHAT YOU SEE — examples below, replace with actual issues]

Example issues to paste in:
- "Player names are being merged into one string instead of separated"
- "Inning numbers are all wrong — looks like diagonal line detection isn't working"
- "The circled plays (run scored) are not being detected"
- "Substitution players are not being detected, only the first player per slot"
- "The ambiguous_cells array is empty but there are clearly illegible cells"

For each issue, update the system prompt in extract.py and/or the Pydantic
validation in models.py to address it. Re-run the dry-run after each fix
and confirm the issue is resolved before moving to the next one.

---

## Prompt 4 — Full import and stat verification

Use after: dry-run output looks correct

---

1. Run a full import of the test image:
   `python process_game.py <image_path> --review`

2. After import, run:
   `python export_season.py --output test_export.xlsx`

3. Verify the following:
   - All 9 batting order slots have at least one player
   - R (runs) column in the export matches the inning_totals.runs_per_inning sum
   - Any player with summary.R not null: verify R in stats matches summary.R
   - Print a diff table showing extracted summary vs computed stats for each
     player where summary values are available

4. If any stat mismatches exceed 1 (rounding), identify which PAs are causing
   the discrepancy and flag them as needs_review=1

---

## Prompt 5 — Second game import and fuzzy matching test

Use after: first game imported cleanly

---

Import a second scorecard from a different game with the same team.
Some player names will be spelled slightly differently.

1. Run `python process_game.py <image_path_2> --dry-run`
2. Show all fuzzy match decisions made (auto-matched and warned)
3. Run `python manage_players.py --aliases` and walk through alias confirmation
4. After confirming aliases, do the full import
5. Run `python export_season.py --output season_2games.xlsx`
6. Verify that players appearing in both games have correct cumulative G=2

If a player is incorrectly split into two records, demonstrate
`python manage_players.py --merge "Name1" "Name2"` to fix it.

---

## Prompt 6 — Edge cases (use if/when encountered)

Reference prompt — use when specific edge cases come up, not in sequence

---

Fix the following edge cases as they are encountered in real imports:

SUBSTITUTION EDGE CASE:
If a substitute enters in inning X but the original player had a PA
that started before inning X and completed after (diagonal line in
their last cell), assign that PA to the original player, not the sub.

INCOMPLETE GAME EDGE CASE:
If a game ends before inning 9 (e.g. mercy rule, rain), the
runs_per_inning array will be shorter than 9. This is valid —
do not pad with zeros or raise a validation error.

MISSING OPPONENT NAME EDGE CASE:
If the opponent cannot be extracted from the image and no --opponent
flag is given, prompt the user interactively:
"Could not extract opponent name. Enter opponent name (or 'unknown'):"
Do not use null as the opponent — it breaks the unique constraint.

POSITION CHANGE EDGE CASE:
A player may change defensive position mid-game (noted by a new
position number in the lineup). Store only the primary (first) position
in the players table. Add a raw_notes field on the player-game level
if needed, but do not over-engineer this.

ILLEGIBLE DATE EDGE CASE:
If date cannot be extracted and no --date flag given, prompt interactively.
Format must be YYYY-MM-DD. Validate format before accepting.
