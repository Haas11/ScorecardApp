"""Per-player data: season/game rows plus a percentile rating scale (#24).

Ratings are a percentile rank among ELIGIBLE players only, per stat: best = 100,
worst = 1, median = 50 (ties share their average rank); K% is inverted since
lower is better there. Missing values and non-eligible players get <NA>.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd

import season_table

RATED_STATS: list[str] = ["AVG", "OBP", "SLG", "OPS", "wOBA", "ISO", "BABIP", "BB%", "K%", "BB/K", "OPS+"]
LOWER_BETTER: set[str] = {"K%"}


@dataclass
class Season:
    players: pd.DataFrame  # Season Stats rows without Team, index 0..n-1, extra bool column "eligible"
    team: pd.Series        # the Team row
    games: pd.DataFrame    # Game Log sheet, column "Date" parsed to datetime64, sorted by Date


def load(raw: bytes) -> Season:
    """Parse the published workbook into a Season."""
    df, small = season_table.load(raw)
    is_team = df["Name"] == "Team"
    team = df[is_team].iloc[0]
    players = df[~is_team].reset_index(drop=True)
    players["eligible"] = ~pd.Series(small, index=df.index)[~is_team].reset_index(drop=True)

    games = pd.read_excel(io.BytesIO(raw), sheet_name="Game Log")
    games["Date"] = pd.to_datetime(games["Date"])
    games = games.sort_values("Date").reset_index(drop=True)
    return Season(players=players, team=team, games=games)


def ratings(players: pd.DataFrame) -> pd.DataFrame:
    """Percentile rating (1-100, nullable Int64) per RATED_STATS column, indexed like players."""
    out = pd.DataFrame(index=players.index, columns=RATED_STATS, dtype="Int64")
    eligible = players["eligible"].fillna(False)
    for stat in RATED_STATS:
        if stat not in players:
            continue
        vals = pd.to_numeric(players[stat], errors="coerce")
        pool = vals[eligible].dropna()
        n = len(pool)
        if n == 0:
            continue
        ascending = stat not in LOWER_BETTER
        ranks = pool.rank(method="average", ascending=ascending)
        if n == 1:
            rating = pd.Series(100, index=pool.index)
        else:
            rating = (1 + 99 * (ranks - 1) / (n - 1)).round()
        out.loc[rating.index, stat] = rating.astype("Int64")
    return out


def player_games(games: pd.DataFrame, name: str) -> pd.DataFrame:
    """That player's Game Log rows by date, plus cum_AB, cum_H, cum_AVG."""
    rows = games[games["Name"] == name].sort_values("Date").reset_index(drop=True)
    rows = rows.copy()
    rows["cum_AB"] = rows["AB"].cumsum()
    rows["cum_H"] = rows["H"].cumsum()
    rows["cum_AVG"] = (rows["cum_H"] / rows["cum_AB"]).where(rows["cum_AB"] != 0)
    return rows


def slug(name: str) -> str:
    """"T. Kryston" -> "t-kryston"."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
