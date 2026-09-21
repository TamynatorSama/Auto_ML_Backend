"""
keys.py
-------
The workspace's model key, decrypted for one task (docs/PHASE5.md §4).

The plaintext exists only in the worker's memory, for as long as the task runs:
never in graph state, never in RunContext, never in a sandbox. The checkpoint
tables and context.json would both keep a copy, and generated code would read
one.
"""

from __future__ import annotations

from db import crypto

PROVIDER = "gemini"   # the only one until another is built (A2)


class MissingKey(RuntimeError):
    """The workspace has no key for the provider this run needs."""


def for_workspace(conn, secret_key: str, workspace_id: int, provider: str = PROVIDER) -> str:
    row = conn.execute(
        "SELECT ciphertext FROM provider_keys WHERE workspace_id = %s AND provider = %s",
        (workspace_id, provider),
    ).fetchone()
    if row is None:
        raise MissingKey(f"this workspace has no {provider} key; add one under Settings, then resume the run")
    return crypto.decrypt(secret_key, row["ciphertext"])
