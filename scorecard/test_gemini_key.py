#!/usr/bin/env python3
"""Quick sanity-check: can we reach the Gemini API with ADC credentials?"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import google.auth
from google import genai

print(f"SDK version: {genai.__version__}")

# Use Application Default Credentials (set up via gcloud auth application-default login)
print("\nLoading Application Default Credentials...")
try:
    creds, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    print(f"OK — project: {project}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

client = genai.Client(credentials=creds)

# List models
print("\nListing models...")
try:
    models = list(client.models.list())
    flash = [m.name for m in models if "flash" in m.name.lower()]
    print(f"OK — {len(models)} models, flash variants: {flash[:4]}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

# Text generation
print("\nText generation test...")
try:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with only the word PASS",
    )
    print(f"Response: {resp.text.strip()}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\nAll checks passed — ADC auth works.")
