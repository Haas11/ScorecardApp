import io

import pandas as pd
import streamlit as st

import data

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

df = pd.read_excel(io.BytesIO(raw), sheet_name="Season Stats")
team = df[df["Name"] == "Team"]
players = df[df["Name"] != "Team"].reset_index(drop=True)

if not team.empty:
    t = team.iloc[0]
    with st.container(horizontal=True):
        st.metric("Games", int(t["G"]), border=True)
        for col in ("AVG", "OBP", "SLG", "OPS"):
            st.metric(f"Team {col}", f"{t[col]:.3f}".lstrip("0"), border=True)
        st.metric("Runs", int(t["R"]), border=True)

rate = st.column_config.NumberColumn(format="%.3f")
st.dataframe(
    players,
    hide_index=True,
    height="content",
    column_config={
        "Name": st.column_config.TextColumn(pinned=True),
        **{c: rate for c in ("AVG", "OBP", "SLG", "OPS", "BABIP", "ISO", "wOBA", "BB%", "K%", "BB/K")
           if c in players},
        "RC": st.column_config.NumberColumn(format="%.1f"),
        "OPS+": st.column_config.NumberColumn(format="%d"),
        "AB/HR": st.column_config.NumberColumn(format="%.1f"),
    },
)
st.caption("Click a column header to sort. Players under the small-sample threshold are listed last.")

st.download_button("Download workbook", raw, file_name=book["name"], icon=":material/download:",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
