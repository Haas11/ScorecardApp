"""One page per game: the published scorecard widget."""
from __future__ import annotations

import streamlit as st

import data


def themed(html: str) -> str:
    """Follow the app's theme instead of the OS: the widget switches to its dark
    palette with @media (prefers-color-scheme: dark), so turn that block on or off."""
    dark = st.context.theme.type == "dark"
    return html.replace("@media (prefers-color-scheme: dark)", "@media all" if dark else "@media not all")


def game_page(game: dict):
    def page() -> None:
        st.title(f"{game['opponent']} ({game['side']})")
        st.caption(f"{game['date']:%A %d %B %Y}")
        # The widget is our own generated HTML from the published folder.
        st.iframe(themed(data.read(game).decode("utf-8")), height="content")

    page.__name__ = f"game_{game['slug'].replace('-', '_')}"  # unique page identity
    return page
