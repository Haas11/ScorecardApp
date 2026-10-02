import streamlit as st

import data
import season_table

st.title("Season stats")

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
rate = st.column_config.NumberColumn(format="%.3f")
st.dataframe(
    df.style.apply(lambda _: season_table.styles(df, small), axis=None),
    hide_index=True,
    height="content",
    column_config={
        "Name": st.column_config.TextColumn(pinned=True),
        **{c: rate for c in ("AVG", "OBP", "SLG", "OPS", "BABIP", "ISO", "wOBA", "BB%", "K%", "BB/K")
           if c in df},
        "RC": st.column_config.NumberColumn(format="%.1f"),
        "OPS+": st.column_config.NumberColumn(format="%d"),
        "AB/HR": st.column_config.NumberColumn(format="%.1f"),
    },
)
st.caption("Grey row: team totals. Amber: season leader. Colour scale: blue below the team average, "
           "red above it (reversed for K% and AB/HR). Players with too few plate appearances are "
           "listed last and not coloured. Click a column header to sort.")

st.download_button("Download workbook", raw, file_name=book["name"], icon=":material/download:",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
