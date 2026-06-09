# Baseball Scorecard Digitizer

A CLI pipeline that photographs a handwritten Dutch KNBSB baseball scorecard, extracts play-by-play data via the Claude vision API, stores results in SQLite, and exports season stats to Excel.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
# Install dependencies
uv sync

# Copy environment template and fill in your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Process a scorecard image

```bash
# Basic usage
uv run python process_game.py path/to/scorecard.jpg

# Override extracted fields
uv run python process_game.py scorecard.jpg --opponent "Giants" --date 2025-05-20

# Validate without writing to DB
uv run python process_game.py scorecard.jpg --dry-run

# Review all PAs before confirming import
uv run python process_game.py scorecard.jpg --review

# Use Opus for dense/messy handwriting
uv run python process_game.py scorecard.jpg --model claude-opus-4-20250514
```

### Review low-confidence plate appearances

```bash
# Review all pending items
uv run python review.py

# Filter by game date or player
uv run python review.py --game 2025-05-20
uv run python review.py --player "Smith"
```

### Manage players and aliases

```bash
# Review fuzzy-matched names from last import
uv run python manage_players.py aliases

# List all players and aliases
uv run python manage_players.py list

# Manually merge two player records
uv run python manage_players.py merge "John Smith" "J. Smith"
```

### Export season stats to Excel

```bash
# Export all players
uv run python export_season.py

# Custom output path and minimum PA filter
uv run python export_season.py --output season_2025.xlsx --min-pa 10
```

## Fuzzy name matching

When a name arrives from extraction it is matched against existing players in this order:

1. **Exact** match on normalized name
2. **Exact** match on stored aliases
3. **Fuzzy** match using `token_sort_ratio`:
   - Score ≥ `auto_match_threshold` (default 90): silent auto-match, logged to console
   - Score ≥ `warn_threshold` (default 70) and < 90: match with warning, queued for review
   - Score < 70: inserted as a new player

Tune thresholds in `config.yml` under the `fuzzy:` key. Lower `warn_threshold` to catch more potential duplicates; raise `auto_match_threshold` to require more confidence before silent matching.

After each import, run `python manage_players.py aliases` to confirm or reject queued fuzzy matches.

## Model selection

| Scorecard quality | Recommended model |
|---|---|
| Clean, printed-style handwriting | `claude-sonnet-4-20250514` (default) |
| Dense, messy, or faded handwriting | `claude-opus-4-20250514` |

Override per-run with `--model` or permanently in `.env`:

```
EXTRACTION_MODEL=claude-opus-4-20250514
```

## Output

- **data/season.db** — SQLite database with all games, PAs, and player records
- **data/raw/** — Archived raw VLM JSON responses, one file per game
- **stats.xlsx** — Season stats workbook with three sheets:
  - *Season Stats*: one row per player, sorted by OPS
  - *Game Log*: per-player per-game breakdown
  - *Low Confidence*: all PAs flagged for review
