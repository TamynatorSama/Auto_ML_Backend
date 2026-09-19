"""
tenancy.py
----------
Every workspace route hangs off `router`, under /api/w/{ws}. Its dependency
resolves the slug and your membership: a workspace you don't belong to answers
404, the same as one that doesn't exist. Routes then use the resolved row's
id, never an id from the request (tests/api/test_tenancy.py walks them all).
"""

from fastapi import APIRouter, Depends, HTTPException

from api.sessions import current_user, get_conn


def workspace(ws: str, user: dict = Depends(current_user), conn=Depends(get_conn)) -> dict:
    row = conn.execute(
        "SELECT w.id, w.slug, w.name, w.created_at, m.role FROM workspaces w "
        "JOIN memberships m ON m.workspace_id = w.id WHERE w.slug = %s AND m.user_id = %s",
        (ws, user["id"]),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Not found")
    return row


router = APIRouter(prefix="/api/w/{ws}", dependencies=[Depends(workspace)])
