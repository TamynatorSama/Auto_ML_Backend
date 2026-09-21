"""
keys.py
-------
The workspace's model key (docs/PHASE5.md §5). Bring your own (D1): the key is
verified against the provider before it is stored, kept as Fernet ciphertext,
and never sent back — the screen only ever sees its last four characters.

    GET    …/providers             which providers have a key, and which are ready at all
    PUT    …/providers/{provider}  verify, then store
    DELETE …/providers/{provider}  forget it

Only Gemini works today (A2); the rest are listed so Settings can grey them out
in their real place rather than pretend they aren't coming.
"""

from __future__ import annotations

import httpx
from fastapi import Depends, HTTPException
from pydantic import BaseModel

from api import tenancy
from api.sessions import current_user, get_conn
from db import crypto

router = tenancy.router

# id -> whether a key for it is any use yet. The names and placeholders are the
# web app's business; these are the ones the canvas lists.
PROVIDERS = {"gemini": True, "anthropic": False, "openai": False, "mistral": False, "xai": False}
GEMINI_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"
VERIFY_TIMEOUT = 20


class KeyIn(BaseModel):
    key: str


def _public(row: dict | None, provider: str) -> dict:
    return {"provider": provider, "available": PROVIDERS[provider],
            "last4": row and row["last4"], "verified_at": row and row["verified_at"]}


def _known(provider: str) -> str:
    if provider not in PROVIDERS:
        raise HTTPException(404, "Not found")
    return provider


def verify(provider: str, key: str) -> None:
    """One real call to the provider. An unverified key would fail inside a plan task instead,
    where the person can't see why (§8.6)."""
    from utils.reusable.hooks import MODEL   # the model the run itself will use

    try:
        response = httpx.post(
            f"{GEMINI_MODELS}/{MODEL}:generateContent",
            headers={"x-goog-api-key": key},
            json={"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}},
            timeout=VERIFY_TIMEOUT,
        )
    except httpx.HTTPError:
        raise HTTPException(503, "The provider couldn't be reached to check that key; try again in a moment")
    if response.status_code == 200:
        return
    raise HTTPException(400, f"The provider refused that key: {_said(response, key)}")


def _said(response: httpx.Response, key: str) -> str:
    """What the provider said, trimmed, with the key taken back out of it.

    Google's refusals don't quote the key, but the message is theirs and we repeat it to
    the browser, so it is scrubbed before it is trimmed — truncating first could leave
    half a key behind.
    """
    try:
        message = (response.json().get("error") or {}).get("message")
    except ValueError:
        message = None
    message = message or f"HTTP {response.status_code}"
    return (message.replace(key, "...") if key else message)[:300]


@router.get("/providers")
def listing(workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    saved = {row["provider"]: row for row in conn.execute(
        "SELECT provider, last4, verified_at FROM provider_keys WHERE workspace_id = %s", (workspace["id"],)
    )}
    return {"providers": [_public(saved.get(provider), provider) for provider in PROVIDERS]}


@router.put("/providers/{provider}")
def save(provider: str, body: KeyIn, workspace: dict = Depends(tenancy.workspace),
         user: dict = Depends(current_user), conn=Depends(get_conn)):
    provider = _known(provider)
    if not PROVIDERS[provider]:
        raise HTTPException(409, "That provider isn't supported yet")
    key = body.key.strip()
    if not key:
        raise HTTPException(422, "Paste the key")
    verify(provider, key)
    row = conn.execute(
        """
        INSERT INTO provider_keys (workspace_id, provider, ciphertext, last4, created_by)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (workspace_id, provider) DO UPDATE SET
            ciphertext = EXCLUDED.ciphertext, last4 = EXCLUDED.last4, verified_at = now(),
            created_by = EXCLUDED.created_by
        RETURNING last4, verified_at
        """,
        (workspace["id"], provider, crypto.encrypt(crypto.secret_key(), key), crypto.last4(key), user["id"]),
    ).fetchone()
    return _public(row, provider)


@router.delete("/providers/{provider}")
def remove(provider: str, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    provider = _known(provider)
    conn.execute("DELETE FROM provider_keys WHERE workspace_id = %s AND provider = %s", (workspace["id"], provider))
    return _public(None, provider)
