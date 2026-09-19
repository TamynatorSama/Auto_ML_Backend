"""Settings from .env that more than one module needs."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# the web app's address: links in emails point here, and changes must come from it
APP_ORIGIN = os.environ.get("AUTOML_APP_ORIGIN", "http://localhost:5173").rstrip("/")
SECURE = APP_ORIGIN.startswith("https://")   # Secure cookies everywhere but plain-http localhost


def oauth_state_secret() -> str:
    """Signs the short-lived OAuth state cookie; derived from AUTOML_SECRET_KEY, so .env needs nothing new."""
    key = os.environ.get("AUTOML_SECRET_KEY")
    if not key:
        raise RuntimeError("AUTOML_SECRET_KEY is not set (see worker/crypto.py)")
    return hashlib.sha256(b"automl oauth state|" + key.encode()).hexdigest()
