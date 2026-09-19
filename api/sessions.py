"""
sessions.py
-----------
The session cookie holds a random token; the database keeps only its SHA-256,
so a copy of the database can't sign anyone in. A session lasts 30 days from
last use; last_seen_at (and the cookie) move on at most once an hour.
"""

import hashlib
import secrets
from datetime import timedelta

from fastapi import Depends, HTTPException, Request, Response

from api import config

LIFETIME = timedelta(days=30)
REFRESH = timedelta(hours=1)
# __Host-: over HTTPS the browser then keeps the cookie to this exact host and path /
COOKIE = "__Host-automl_session" if config.SECURE else "automl_session"


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def get_conn(request: Request):
    """A pooled connection for one request (autocommit; use conn.transaction() to group writes)."""
    with request.app.state.pool.connection() as conn:
        yield conn


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(COOKIE, token, max_age=int(LIFETIME.total_seconds()), path="/",
                        httponly=True, samesite="lax", secure=config.SECURE)


def sign_in(conn, response: Response, user_id: int, request: Request | None = None) -> None:
    token = secrets.token_urlsafe(32)
    conn.execute("DELETE FROM sessions WHERE user_id = %s AND expires_at < now()", (user_id,))
    old = request and request.cookies.get(COOKIE)
    if old:   # this browser's last session, whosever it was: signing in again ends it
        conn.execute("DELETE FROM sessions WHERE id = %s", (token_hash(old),))
    conn.execute("INSERT INTO sessions (id, user_id, expires_at) VALUES (%s, %s, now() + %s)",
                 (token_hash(token), user_id, LIFETIME))
    _set_cookie(response, token)


def sign_out(conn, request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE)
    if token:
        conn.execute("DELETE FROM sessions WHERE id = %s", (token_hash(token),))
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="lax", secure=config.SECURE)


def end_all(conn, user_id: int) -> None:
    conn.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))


def current_user(request: Request, response: Response, conn=Depends(get_conn)) -> dict:
    """The signed-in user (id, email, name), else 401."""
    token = request.cookies.get(COOKIE)
    row = token and conn.execute(
        "SELECT u.id, u.email, u.name, s.last_seen_at < now() - %s AS stale "
        "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.id = %s AND s.expires_at > now()",
        (REFRESH, token_hash(token)),
    ).fetchone()
    if not row:
        raise HTTPException(401, "Not signed in")
    if row.pop("stale"):
        conn.execute("UPDATE sessions SET last_seen_at = now(), expires_at = now() + %s WHERE id = %s",
                     (LIFETIME, token_hash(token)))
        _set_cookie(response, token)
    return row
