import pandas as pd
import streamlit as st

import data
import season_table

st.title("Quick H1 2026 Stats")
# Phones: a smaller title so the table starts higher up the screen.
st.html("<style>@media (max-width: 640px) { h1 { font-size: 1.6rem !important; } }</style>")

books = data.workbooks()
if not books:
    st.warning("No season workbook in the published folder yet.", icon=":material/warning:")
    st.caption(data.describe_source())
    st.caption("Files seen: " + (", ".join(f["name"] for f in data.list_files()[:10]) or "none"))
    st.stop()

# Ex Spring Training is the default season; other workbooks can be picked.
default = books.index(data.default_workbook())
book = books[0] if len(books) == 1 else st.selectbox(
    "Season", books, index=default, format_func=lambda b: b["name"].removesuffix(" stats.xlsx"))
raw = data.read(book)

df, small = season_table.load(raw)
css = season_table.styles(df, small)  # on the whole sheet, so averages and scales match the workbook
FORMATS = {**{c: "%.3f" for c in ("AVG", "OBP", "SLG", "OPS", "BABIP", "ISO", "wOBA", "BB%", "K%", "BB/K")},
           "RC": "%.1f", "OPS+": "%d", "AB/HR": "%.1f"}
column_config = {
        "Name": st.column_config.TextColumn(pinned=True),
        **{c: st.column_config.NumberColumn(help=season_table.STAT_HELP.get(c), format=FORMATS.get(c))
           for c in df.columns if c != "Name"},
}

# Column groups, so a phone shows a screen-wide table instead of all 26 columns.
VIEWS = {
    "Overview": ["Name", "PA", "AVG", "OBP", "SLG", "OPS", "H", "HR", "R", "RBI", "SB"],
    "Counting": ["Name", "G", "PA", "AB", "H", "2B", "3B", "HR", "R", "RBI", "BB", "K", "SB"],
    "Advanced": ["Name", "PA", "OPS", "BABIP", "ISO", "BB%", "K%", "wOBA", "RC", "OPS+", "AB/HR", "BB/K"],
    "All": list(df.columns),
}
view = st.segmented_control("Columns", list(VIEWS), default="Overview", key="season_view",
                            label_visibility="collapsed") or "Overview"
columns = [c for c in VIEWS[view] if c in df]
st.caption("Hover on a stat for an explanation. Click a stat to sort on it.")

ROW_PX, HEADER_PX, MAX_PX = 24, 30, 460


def table(rows):
    part = df[rows]
    # A fixed height (not "content") makes the table scroll inside itself, so the
    # header row stays frozen like the pinned Name column. Short tables keep their size.
    height = min(HEADER_PX + ROW_PX * len(part) + 2, MAX_PX)
    st.dataframe(part.style.apply(lambda _: css[rows], axis=None), hide_index=True,
                 height=height, row_height=ROW_PX, column_order=columns, column_config=column_config)


# Players with too few plate appearances get their own table, so sorting the
# main table only ranks eligible players.
eligible = ~pd.Series(small, index=df.index)
table(eligible)
st.caption("Grey row: team totals. Amber: season leader. Colour scale: blue below the team average, "
           "red above it (reversed for K% and AB/HR).")
if not eligible.all():
    st.subheader("Too few plate appearances")
    table(~eligible)

st.download_button("Download workbook", raw, file_name=book["name"], icon=":material/download:",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
