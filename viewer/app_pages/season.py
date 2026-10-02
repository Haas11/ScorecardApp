import pandas as pd
import streamlit as st

import data
import season_table

st.title("Quick H1 2026 Stats")

books = data.workbooks()
if not books:
    st.warning("No season workbook in the published folder yet.", icon=":material/warning:")
    st.caption(data.describe_source())
    st.caption("Files seen: " + (", ".join(f["name"] for f in data.list_files()[:10]) or "none"))
    st.stop()

# Ex Spring Training is the default season; other workbooks can be picked.
default = next((i for i, b in enumerate(books) if "Ex Spring Training" in b["name"]), 0)
book = books[0] if len(books) == 1 else st.selectbox(
    "Season", books, index=default, format_func=lambda b: b["name"].removesuffix(" stats.xlsx"))
raw = data.read(book)

df, small = season_table.load(raw)
css = season_table.styles(df, small)  # on the whole sheet, so averages and scales match the workbook
rate = st.column_config.NumberColumn(format="%.3f")
column_config = {
        "Name": st.column_config.TextColumn(pinned=True),
        **{c: rate for c in ("AVG", "OBP", "SLG", "OPS", "BABIP", "ISO", "wOBA", "BB%", "K%", "BB/K")
           if c in df},
        "RC": st.column_config.NumberColumn(format="%.1f"),
        "OPS+": st.column_config.NumberColumn(format="%d"),
        "AB/HR": st.column_config.NumberColumn(format="%.1f"),
}


def table(rows):
    part = df[rows]
    st.dataframe(part.style.apply(lambda _: css[rows], axis=None),
                 hide_index=True, height="content", column_config=column_config)


# Players with too few plate appearances get their own table, so sorting the
# main table only ranks eligible players.
eligible = ~pd.Series(small, index=df.index)
table(eligible)
st.caption("Grey row: team totals. Amber: season leader. Colour scale: blue below the team average, "
           "red above it (reversed for K% and AB/HR). Click a column header to sort.")
if not eligible.all():
    st.subheader("Too few plate appearances")
    table(~eligible)

st.download_button("Download workbook", raw, file_name=book["name"], icon=":material/download:",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
