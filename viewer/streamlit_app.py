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

try:
    games = data.games()
except Exception as exc:  # Drive login/permission problems: show Google's reason (never contains the key)
    if type(exc).__module__.startswith(("google", "googleapiclient")):
        st.error(f"Google Drive refused the request: {type(exc).__name__}: {exc}", icon=":material/error:")
        st.caption("RefreshError = the service account login failed: every [gcp_service_account] value must "
                   "come from the same, current JSON key file. HttpError 403/404 = the folder isn't shared "
                   "with the service account's email, or folder_id is wrong.")
        st.stop()
    raise
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
