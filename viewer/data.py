"""Where the team viewer gets its files: the published folder (widgets + workbook).

On Streamlit Cloud that is the Google Drive folder, read with a service account
(secrets: [drive] folder_id, [gcp_service_account] key). Locally, without those
secrets, it is the Drive folder as synced to this PC (VIEWER_DATA_DIR, default
G:/My Drive/Quick 2026). The app only reads; publish.py writes the folder.
"""
from __future__ import annotations

import csv
import io
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

LOCAL_DIR_DEFAULT = "G:/My Drive/Quick 2026"
_GAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*-?\s*(.+?)\s*\((Home|Away)\)$")
_ROSTER_COLUMNS = ["name", "number", "position", "bats", "throws", "photo"]


def _secret(section: str) -> dict | None:
    try:
        return dict(st.secrets[section]) if section in st.secrets else None
    except Exception:  # no secrets.toml at all (local run)
        return None


def using_drive() -> bool:
    return bool(_secret("drive") and _secret("gcp_service_account"))


def _drive_service():
    acct = _secret("gcp_service_account")
    return _drive_service_for(acct.get("client_email", ""), acct.get("private_key_id", ""))


@st.cache_resource(max_entries=2)
def _drive_service_for(client_email: str, key_id: str):
    # Keyed on account + key id: edited Secrets give a new login instead of the
    # cached one from the previous key.
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_info(
        _secret("gcp_service_account"), scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _drive_list_children(svc, parent_id: str) -> list[dict]:
    """[{name, key, modified}] of every file directly under a Drive folder id."""
    files, token = [], None
    while True:
        resp = svc.files().list(
            q=f"'{parent_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, modifiedTime)",
            pageSize=1000, pageToken=token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files += [{"name": f["name"], "key": f["id"], "modified": f["modifiedTime"]}
                  for f in resp.get("files", [])]
        token = resp.get("nextPageToken")
        if not token:
            return files


def _drive_find_subfolder(svc, parent_id: str, name: str) -> str | None:
    """The Drive folder id of the child folder `name` directly under parent_id, if any."""
    resp = svc.files().list(
        q=(f"'{parent_id}' in parents and trashed = false and "
           "mimeType = 'application/vnd.google-apps.folder' and "
           f"name = '{name}'"),
        fields="files(id, name)", pageSize=10,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    found = resp.get("files", [])
    return found[0]["id"] if found else None


@st.cache_data(ttl=300, show_spinner=False)
def list_files() -> list[dict]:
    """[{name, key, modified}] of the published folder, refreshed every 5 minutes."""
    if using_drive():
        return _drive_list_children(_drive_service(), _secret("drive")["folder_id"])
    root = Path(os.environ.get("VIEWER_DATA_DIR", LOCAL_DIR_DEFAULT))
    return [{"name": p.name, "key": str(p), "modified": str(p.stat().st_mtime)}
            for p in sorted(root.glob("*")) if p.is_file()]


@st.cache_data(ttl=300, show_spinner=False)
def _list_player_files() -> list[dict]:
    """[{name, key, modified}] inside the published folder's 'players' subfolder
    (one level deep: photos only, never recursed further)."""
    if using_drive():
        svc, folder = _drive_service(), _secret("drive")["folder_id"]
        sub_id = _drive_find_subfolder(svc, folder, "players")
        return _drive_list_children(svc, sub_id) if sub_id else []
    root = Path(os.environ.get("VIEWER_DATA_DIR", LOCAL_DIR_DEFAULT)) / "players"
    if not root.is_dir():
        return []
    return [{"name": p.name, "key": str(p), "modified": str(p.stat().st_mtime)}
            for p in sorted(root.glob("*")) if p.is_file()]


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def _read(key: str, modified: str) -> bytes:  # `modified` is part of the cache key
    if using_drive():
        return _drive_service().files().get_media(fileId=key).execute()
    return Path(key).read_bytes()


def describe_source() -> str:
    """What the app is reading, for the 'nothing found' message."""
    files = list_files()
    if using_drive():
        email = _secret("gcp_service_account").get("client_email", "?")
        hint = ("" if files else " The service account sees nothing there: share the folder with "
                f"{email} (Viewer), and check that folder_id is the ID at the end of the folder's URL.")
        return f"Google Drive folder {_secret('drive')['folder_id']} as {email}: {len(files)} file(s).{hint}"
    root = Path(os.environ.get("VIEWER_DATA_DIR", LOCAL_DIR_DEFAULT))
    secrets = [s for s in ("drive", "gcp_service_account") if not _secret(s)]
    return (f"Local folder {root} ({'exists' if root.is_dir() else 'does not exist'}): "
            f"{len(files)} file(s). Not using Drive because the secrets lack "
            f"{' and '.join(f'[{s}]' for s in secrets)}.")


def read(entry: dict) -> bytes:
    return _read(entry["key"], entry["modified"])


def workbooks() -> list[dict]:
    return [f for f in list_files() if f["name"].endswith(".xlsx") and not f["name"].startswith("~$")]


def default_workbook() -> dict | None:
    """The season workbook to show when nothing else is picked: 'Ex Spring Training' if present."""
    books = workbooks()
    if not books:
        return None
    return next((b for b in books if "Ex Spring Training" in b["name"]), books[0])


def games() -> list[dict]:
    """Published game widgets, newest first, with date / opponent / side parsed from the name."""
    out = []
    for f in list_files():
        m = _GAME_RE.match(f["name"][:-5]) if f["name"].endswith(".html") else None
        if m:
            d = date.fromisoformat(m.group(1))
            out.append({**f, "date": d, "opponent": m.group(2), "side": m.group(3),
                        "slug": re.sub(r"[^a-z0-9]+", "-", f["name"][:-5].lower()).strip("-")})
    return sorted(out, key=lambda g: g["date"], reverse=True)


def _parse_roster_csv(text: str) -> pd.DataFrame:
    """Parse players.csv text into a DataFrame with _ROSTER_COLUMNS, tolerating
    ',' or ';' delimiters (pure function, no I/O — easy to unit test)."""
    lines = text.splitlines()
    if not lines:
        return pd.DataFrame(columns=_ROSTER_COLUMNS)
    try:
        delim = csv.Sniffer().sniff(lines[0], delimiters=",;").delimiter
    except csv.Error:
        delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
    df = pd.read_csv(io.StringIO(text), delimiter=delim, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=lambda c: c.strip())
    for col in df.columns:
        df[col] = df[col].str.strip()
    for col in _ROSTER_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[_ROSTER_COLUMNS]


@st.cache_data(ttl=300, show_spinner=False)
def roster() -> pd.DataFrame:
    """Parsed players.csv from the published folder (name, number, position,
    bats, throws, photo); an empty DataFrame with those columns if absent."""
    entry = next((f for f in list_files() if f["name"] == "players.csv"), None)
    if entry is None:
        return pd.DataFrame(columns=_ROSTER_COLUMNS)
    text = read(entry).decode("utf-8-sig")
    return _parse_roster_csv(text)


def player_photo(filename: str) -> bytes | None:
    """Raw bytes of a photo in the published 'players/' subfolder, or None if
    `filename` is blank or not found there."""
    if not filename:
        return None
    entry = next((f for f in _list_player_files() if f["name"] == filename), None)
    return read(entry) if entry is not None else None
