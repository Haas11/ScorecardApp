"""One page per game: the published scorecard widget."""
from __future__ import annotations

import streamlit as st

import data


def game_page(game: dict):
    def page() -> None:
        st.title(f"{game['opponent']} ({game['side']})")
        st.caption(f"{game['date']:%A %d %B %Y}")
        # The widget is our own generated HTML from the published folder.
        st.iframe(data.read(game).decode("utf-8"), height="content")

    page.__name__ = f"game_{game['slug'].replace('-', '_')}"  # unique page identity
    return page
