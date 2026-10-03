"""Per-player page (#25): header, bio card, rating bars, cumulative AVG trend."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import data
import player_bars
import player_trend
import players
import season_table


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def _load(key: str, modified: str) -> players.Season:
    return players.load(data.read({"key": key, "modified": modified}))


def current_season() -> players.Season | None:
    """The default workbook's Season, cached on (key, modified) -- not on the raw bytes."""
    book = data.default_workbook()
    return _load(book["key"], book["modified"]) if book else None



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

        info = {}
        roster = data.roster()
        if not roster.empty:
            match = roster[roster["name"] == name]
            if not match.empty:
                info = {k: v for k, v in match.iloc[0].to_dict().items() if pd.notna(v) and str(v).strip()}

        left, mid, right = st.columns([1, 1.4, 1.2])

        # Left: photo (if any), name, slash line, then the player details.
        with left:
            with st.container(border=True):
                photo = data.player_photo(info.get("photo", "")) if info.get("photo") else None
                if photo:
                    st.image(photo)
                st.header(name, anchor=False, divider="gray")
                st.subheader(f"{_fmt_rate(row['AVG'])} / {_fmt_rate(row['OBP'])} / {_fmt_rate(row['SLG'])}",
                             anchor=False)
                st.caption(f"AVG / OBP / SLG · {int(row['G'])} G · {int(row['PA'])} PA",
                           help="\n\n".join(f"**{k}**: {season_table.STAT_HELP[k]}"
                                            for k in ("AVG", "OBP", "SLG", "G", "PA")))
                details = [f"#{info['number']}" if info.get("number") else None,
                           info.get("position"),
                           f"B/T: {info.get('bats', '-')}/{info.get('throws', '-')}"
                           if info.get("bats") or info.get("throws") else None]
                details = [d for d in details if d]
                st.markdown(" · ".join(details) if details else ":gray[No player details yet.]")

        with mid, st.container(border=True):
            if row["eligible"]:
                values = {stat: (float(row[stat]) if pd.notna(row[stat]) else None)
                          for stat in players.RATED_STATS if stat in row}
                rdf = players.ratings(season.players)
                ratings_row = {stat: (int(rdf.at[row.name, stat]) if pd.notna(rdf.at[row.name, stat]) else None)
                               for stat in players.RATED_STATS if stat in rdf}
                player_bars.render(values, ratings_row)
            else:
                st.info("Too few plate appearances for ratings.", icon=":material/info:")

        with right, st.container(border=True):
            games = players.player_games(season.games, name)
            player_trend.render(games, float(season.team["AVG"]))

    page.__name__ = f"player_{players.slug(name)}"  # unique page identity
    return page
