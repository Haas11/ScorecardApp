"""Check a service-account JSON key against the Drive folder, locally.

    uv run --with google-auth --with google-api-python-client python viewer/check_drive_key.py KEY.json [FOLDER_ID]

Prints the account, the key id and the files the account can see. The key stays
on this PC; nothing is uploaded except the normal Google login.
"""
import json
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build

key_path = sys.argv[1]
folder = sys.argv[2] if len(sys.argv) > 2 else "1Bo1orMP9Tlbg62X9cugleZOeXk_oHiSV"
info = json.load(open(key_path, encoding="utf-8"))
print(f"account: {info.get('client_email')}\nkey id : {info.get('private_key_id')}")
creds = service_account.Credentials.from_service_account_info(
    info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
svc = build("drive", "v3", credentials=creds, cache_discovery=False)
files = svc.files().list(q=f"'{folder}' in parents and trashed = false",
                         fields="files(name)", pageSize=50,
                         supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get("files", [])
print(f"login OK, {len(files)} file(s) visible in the folder")
for f in files[:10]:
    print("  ", f["name"])
