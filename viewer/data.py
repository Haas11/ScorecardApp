"""Where the team viewer gets its files: the published folder (widgets + workbook).

On Streamlit Cloud that is the Google Drive folder, read with a service account
(secrets: [drive] folder_id, [gcp_service_account] key). Locally, without those
secrets, it is the Drive folder as synced to this PC (VIEWER_DATA_DIR, default
G:/My Drive/Quick 2026). The app only reads; publish.py writes the folder.
"""
from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import streamlit as st

LOCAL_DIR_DEFAULT = "G:/My Drive/Quick 2026"
_GAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*-?\s*(.+?)\s*\((Home|Away)\)$")


def _secret(section: str) -> dict | None:
    try:
        return dict(st.secrets[section]) if section in st.secrets else None
    except Exception:  # no secrets.toml at all (local run)
        return None


def using_drive() -> bool:
    return bool(_secret("drive") and _secret("gcp_service_account"))


@st.cache_resource
def _drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_info(
        _secret("gcp_service_account"), scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


@st.cache_data(ttl=300, show_spinner=False)
def list_files() -> list[dict]:
    """[{name, key, modified}] of the published folder, refreshed every 5 minutes."""
    if using_drive():
        svc, folder = _drive_service(), _secret("drive")["folder_id"]
        files, token = [], None
        while True:
            resp = svc.files().list(
                q=f"'{folder}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, modifiedTime)",
                pageSize=1000, pageToken=token,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            files += [{"name": f["name"], "key": f["id"], "modified": f["modifiedTime"]}
                      for f in resp.get("files", [])]
            token = resp.get("nextPageToken")
            if not token:
                return files
    root = Path(os.environ.get("VIEWER_DATA_DIR", LOCAL_DIR_DEFAULT))
    return [{"name": p.name, "key": str(p), "modified": str(p.stat().st_mtime)}
            for p in sorted(root.glob("*")) if p.is_file()]


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def _read(key: str, modified: str) -> bytes:  # `modified` is part of the cache key
    if using_drive():
        return _drive_service().files().get_media(fileId=key).execute()
    return Path(key).read_bytes()


def read(entry: dict) -> bytes:
    return _read(entry["key"], entry["modified"])


def workbooks() -> list[dict]:
    return [f for f in list_files() if f["name"].endswith(".xlsx") and not f["name"].startswith("~$")]


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
