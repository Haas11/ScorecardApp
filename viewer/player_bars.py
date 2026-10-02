"""Middle column: Savant-style percentile rating bars (#26).

One row per stat in players.RATED_STATS order: a grey 0-100 track, a bar
filled to the player's percentile rating and coloured on the blue (1) -
white (50) - red (100) heat-map scale, a circle with the rating number at
its tip, and the player's actual stat value printed to the right. A stat
with no rating (small-sample teammates pool, or a missing value) still
shows its track and value, just no bar/circle.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import players
import season_table

_X_DOMAIN = [0, 107]   # bars run 0-100; a little room so the circle at 100 isn't clipped
_VALUE_GAP = 70        # px between the stat name and the bars, where the value is drawn

_PCT_STATS = {"BB%", "K%"}
_RATE_STATS = {"AVG", "OBP", "SLG", "OPS", "wOBA", "ISO", "BABIP"}


def rating_color(rating: int) -> str:
    """Heat-map colour for a 1-100 rating: 1 -> BLUE, 50 -> WHITE, 100 -> RED."""
    r = min(max(rating, 1), 100)
    if r <= 50:
        return season_table._mix(season_table.BLUE, season_table.WHITE, (r - 1) / 49)
    return season_table._mix(season_table.WHITE, season_table.RED, (r - 50) / 50)


def format_value(stat: str, v: float | None) -> str:
    """Stat-specific display text; '-' for a missing value."""
    if v is None or pd.isna(v):
        return "-"
    if stat in _RATE_STATS:
        return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"
    if stat in _PCT_STATS:
        return f"{v * 100:.1f}%"
    if stat == "BB/K":
        return f"{v:.2f}"
    if stat == "OPS+":
        return f"{v:.0f}"
    return f"{v:.3f}"


def _rows(values: dict[str, float | None], ratings_row: dict[str, int | None]) -> list[dict]:
    rows = []
    for stat in players.RATED_STATS:
        v = values.get(stat)
        r = ratings_row.get(stat)
        if v is None and r is None:
            continue
        rows.append({
            "stat": stat,
            "value_text": format_value(stat, v),
            "rating": float(r) if r is not None else None,
            "rating_label": str(int(r)) if r is not None else None,
            "color": rating_color(r) if r is not None else None,
            "num_color": "white" if (r is not None and (r <= 25 or r >= 75)) else "black",
        })
    return rows


def render(values: dict[str, float | None], ratings_row: dict[str, int | None]) -> None:
    """Draw the percentile-rating bars for one player's rated stats."""
    st.markdown("**Percentile rankings**")
    st.caption("Versus eligible teammates: 100 = best, 50 = median, 1 = worst.")

    rows = _rows(values, ratings_row)
    if not rows:
        st.caption("No rated stats available.")
        return

    df = pd.DataFrame(rows)
    stat_order = df["stat"].tolist()
    df["x0"], df["x1"], df["zero"] = 0, 100, 0
    bar_df = df[df["rating"].notna()].copy()

    y_enc = alt.Y(
        "stat:N",
        sort=stat_order,
        # labelPadding leaves room between the stat name and the bars for the value text.
        axis=alt.Axis(title=None, labelFontSize=15, labelPadding=_VALUE_GAP, grid=False,
                      domain=False, ticks=False),
    )
    tooltip = [
        alt.Tooltip("stat:N", title="Stat"),
        alt.Tooltip("value_text:N", title="Value"),
        alt.Tooltip("rating_label:N", title="Rating"),
    ]

    def hidden_x(field: str) -> alt.X:
        return alt.X(f"{field}:Q", scale=alt.Scale(domain=_X_DOMAIN), axis=None)

    track = alt.Chart(df).mark_bar(size=6, cornerRadius=3, color="#E6E6E6").encode(
        x=hidden_x("x0"), x2="x1:Q", y=y_enc,
    )
    bar = alt.Chart(bar_df).mark_bar(size=10, cornerRadius=5).encode(
        x=hidden_x("zero"), x2="rating:Q", y=y_enc,
        color=alt.Color("color:N", scale=None),
        tooltip=tooltip,
    )
    circle = alt.Chart(bar_df).mark_circle(size=760, opacity=1, stroke="white", strokeWidth=2).encode(
        x=hidden_x("rating"), y=y_enc,
        color=alt.Color("color:N", scale=None),
        tooltip=tooltip,
    )
    number = alt.Chart(bar_df).mark_text(fontWeight="bold", fontSize=13).encode(
        x=hidden_x("rating"), y=y_enc,
        text="rating_label:N",
        color=alt.Color("num_color:N", scale=None),
    )
    value_text = alt.Chart(df).mark_text(align="right", dx=-14, fontSize=15).encode(
        x=hidden_x("x0"), y=y_enc,
        text="value_text:N",
    )

    chart = (
        alt.layer(track, bar, circle, number, value_text)
        .properties(height=38 * len(df), background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")
