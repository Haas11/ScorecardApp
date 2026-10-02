"""Right column: cumulative AVG over the season (#27).

Altair line of cumulative AVG by game date, with per-game point markers and a
dashed horizontal rule at the team season AVG.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def _fmt_avg(v: float) -> str:
    """.605 / 1.000 style: three decimals, no leading zero."""
    return f"{v:.3f}".lstrip("0")


def render(games: pd.DataFrame, team_avg: float) -> None:
    """Draw the cumulative-AVG trend line plus the team-AVG reference rule."""
    st.markdown("**Batting average over the season**")
    st.caption("Cumulative AVG after each game. Dashed line: team AVG.")

    valid = games[games["cum_AVG"].notna()].copy()
    if len(valid) < 2:
        st.caption("Not enough games with at-bats yet.")
        return

    valid["_date_str"] = valid["Date"].dt.day.astype(str) + " " + valid["Date"].dt.strftime("%b")
    valid["_game_line"] = valid["H"].astype(int).astype(str) + "-" + valid["AB"].astype(int).astype(str)
    valid["_avg_str"] = valid["cum_AVG"].apply(_fmt_avg)

    vals = list(valid["cum_AVG"]) + [team_avg]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = max(span * 0.15, 0.01)
    domain = [lo - pad, hi + pad]

    y_scale = alt.Scale(domain=domain, zero=False)

    line = (
        alt.Chart(valid)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%d %b", title=None)),
            y=alt.Y(
                "cum_AVG:Q",
                scale=y_scale,
                axis=alt.Axis(title="AVG", labelExpr="replace(format(datum.value, '.3f'), /^0/, '')"),
            ),
            tooltip=[
                alt.Tooltip("_date_str:N", title="Date"),
                alt.Tooltip("Opponent:N", title="Opponent"),
                alt.Tooltip("_game_line:N", title="H-AB"),
                alt.Tooltip("_avg_str:N", title="AVG"),
            ],
        )
    )

    rule_df = pd.DataFrame({"y": [team_avg]})
    rule = (
        alt.Chart(rule_df)
        .mark_rule(strokeDash=[6, 4], color="grey")
        .encode(y=alt.Y("y:Q", scale=y_scale))
    )

    label_df = pd.DataFrame({
        "x": [valid["Date"].max()],
        "y": [team_avg],
        "label": [f"Team {_fmt_avg(team_avg)}"],
    })
    label = (
        alt.Chart(label_df)
        .mark_text(align="left", dx=6, dy=-6, color="grey", fontSize=13)
        .encode(x="x:T", y=alt.Y("y:Q", scale=y_scale), text="label:N")
    )

    chart = (
        alt.layer(rule, line, label)
        .properties(height=300, width="container", background="transparent")
        .configure_axis(labelFontSize=13, titleFontSize=14)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")
