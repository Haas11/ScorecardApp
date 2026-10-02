"""Per-player page (#25): header, bio card, rating bars, cumulative AVG trend."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

import data
import player_bars
import player_trend
import players


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _load(key: str, modified: str) -> players.Season:
    return players.load(data.read({"key": key, "modified": modified}))


def current_season() -> players.Season | None:
    """The default workbook's Season, cached on (key, modified) -- not on the raw bytes."""
    book = data.default_workbook()
    return _load(book["key"], book["modified"]) if book else None


def _initials(name: str) -> str:
    tokens = [t for t in re.split(r"[\s.]+", name) if t]
    if len(tokens) > 1:
        return (tokens[0][0] + tokens[-1][0]).upper()
    return tokens[0][:2].upper()


def _fmt_rate(v: float) -> str:
    """.333 / 1.000 style: three decimals, no leading zero."""
    return f"{v:.3f}".lstrip("0") if pd.notna(v) else "-"


def player_page(name: str):
    def page() -> None:
        season = current_season()
        if season is None:
            st.error("No published workbook found.", icon=":material/error:")
            st.stop()
        rows = season.players[season.players["Name"] == name]
        if rows.empty:
            st.error(f"No stats found for {name}.", icon=":material/error:")
            st.stop()
        row = rows.iloc[0]

        st.title(name)
        st.caption(f"G {int(row['G'])} · PA {int(row['PA'])} · "
                   f"{_fmt_rate(row['AVG'])}/{_fmt_rate(row['OBP'])}/{_fmt_rate(row['SLG'])}")

        left, mid, right = st.columns([1, 1.4, 1.2])

        with left:
            with st.container(border=True):
                info = {}
                roster = data.roster()
                if not roster.empty:
                    match = roster[roster["name"] == name]
                    if not match.empty:
                        info = match.iloc[0].to_dict()

                photo = data.player_photo(info.get("photo", "")) if info else None
                if photo:
                    st.image(photo)
                else:
                    st.header(_initials(name))
                st.write(name)

                lines = []
                if info.get("number"):
                    lines.append(f"#{info['number']}")
                if info.get("position"):
                    lines.append(info["position"])
                if info.get("bats") and info.get("throws"):
                    lines.append(f"B/T: {info['bats']}/{info['throws']}")
                elif info.get("bats"):
                    lines.append(f"Bats: {info['bats']}")
                elif info.get("throws"):
                    lines.append(f"Throws: {info['throws']}")
                if lines:
                    for line in lines:
                        st.caption(line)
                else:
                    st.caption("No player details yet.")

        with mid:
            if row["eligible"]:
                values = {stat: (float(row[stat]) if pd.notna(row[stat]) else None)
                          for stat in players.RATED_STATS if stat in row}
                rdf = players.ratings(season.players)
                ratings_row = {stat: (int(rdf.at[row.name, stat]) if pd.notna(rdf.at[row.name, stat]) else None)
                               for stat in players.RATED_STATS if stat in rdf}
                player_bars.render(values, ratings_row)
            else:
                st.info("Too few plate appearances for ratings.", icon=":material/info:")

        with right:
            games = players.player_games(season.games, name)
            player_trend.render(games, float(season.team["AVG"]))

    page.__name__ = f"player_{players.slug(name)}"  # unique page identity
    return page
