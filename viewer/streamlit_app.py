"""Team viewer: season stats and one page per game, read-only.

    cd viewer
    uv run --with-requirements requirements.txt streamlit run streamlit_app.py

Data comes from the published folder (see data.py). Processing games happens in
the pipeline (scorecard/), which publishes here with `scorecard.py publish`.
"""
from __future__ import annotations

import streamlit as st

import data
from game_view import game_page

st.set_page_config(page_title="Quick stats", page_icon=":material/sports_baseball:", layout="wide")

games = data.games()
pages = {
    "": [st.Page("app_pages/season.py", title="Season stats", icon=":material/leaderboard:", default=True)],
    "Games": [
        st.Page(game_page(g), title=f"{g['date']:%d %b} · {g['opponent']} ({g['side']})",
                icon=":material/sports_baseball:", url_path=g["slug"])
        for g in games
    ],
}
page = st.navigation(pages, position="sidebar")
page.run()
