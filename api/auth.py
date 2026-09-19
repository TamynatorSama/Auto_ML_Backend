"""
auth.py
-------
Password sign-up, email verification, log-in, log-out and forgotten passwords
(docs/PHASE3.md §4). Emailed links carry a random token, stored as its
SHA-256, that works once. Emails are sent after the response, so answering
never takes longer for an address that has an account.
"""

import math
import re
import secrets
from datetime import timedelta

import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import config, mail, sessions
from api.sessions import get_conn
from api.workspaces import TAKEN, check_name, check_slug, create_workspace, me

router = APIRouter(prefix="/api/auth")
hasher = PasswordHasher()   # argon2id
DUMMY_HASH = hasher.hash("no such account")   # checked when there's no account, so that costs the same

EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
LINK_LIFETIME = {"verify": timedelta(hours=24), "reset": timedelta(hours=1)}
RESEND_AFTER = timedelta(minutes=1)   # at most one email of each kind a minute per person
MAX_FAILURES, FAILURE_WINDOW = 5, timedelta(minutes=15)

WRONG = "Email or password is wrong"
BAD_LINK = "This link is invalid or has expired"
NOT_INVITED = "This email isn't invited to the beta yet"
EXISTS = "An account with this email already exists. Log in instead."
FORGOT_SENT = "If there's an account for that email, we've sent a link"


class SignupIn(BaseModel):
    name: str = Field(max_length=100)
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)
    workspace: str = Field(max_length=64)


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class EmailIn(BaseModel):
    email: str = Field(max_length=254)


class TokenIn(BaseModel):
    token: str = Field(max_length=128)


class ResetIn(TokenIn):
    password: str = Field(max_length=256)


# ---- building blocks (oauth.py uses some of them)

def normal_email(text: str) -> str:
    return text.strip().lower()


def check_password(password: str) -> None:
    if len(password) < 10:
        raise HTTPException(422, "Use at least 10 characters for your password")


def invited(conn, email: str) -> bool:
    return conn.execute("SELECT 1 FROM signup_allowlist WHERE email = %s", (email,)).fetchone() is not None


def password_ok(user: dict | None, password: str) -> bool:
    stored = user["password_hash"] if user else None
    try:
        hasher.verify(stored or DUMMY_HASH, password)
    except (VerificationError, InvalidHashError):
        return False
    return stored is not None


def _email_link(pool, user_id: int, purpose: str) -> None:
    """Runs after the response: a fresh token, unless one of this kind went out in the last minute."""
    token = secrets.token_urlsafe(32)
    with pool.connection() as conn, conn.transaction():
        if conn.execute("SELECT 1 FROM email_tokens WHERE user_id = %s AND purpose = %s AND created_at > now() - %s",
                        (user_id, purpose, RESEND_AFTER)).fetchone():
            return
        conn.execute("DELETE FROM email_tokens WHERE user_id = %s AND (used_at IS NOT NULL OR expires_at < now())",
                     (user_id,))
        conn.execute("INSERT INTO email_tokens (id, user_id, purpose, expires_at) VALUES (%s, %s, %s, now() + %s)",
                     (sessions.token_hash(token), user_id, purpose, LINK_LIFETIME[purpose]))
        user = conn.execute("SELECT email, name FROM users WHERE id = %s", (user_id,)).fetchone()
    # in the fragment, not the query: a fragment never reaches a server or a Referer header
    mail.send_link(user["email"], user["name"], purpose, f"{config.APP_ORIGIN}/{purpose}#token={token}")


def email_link(request: Request, background: BackgroundTasks, user_id: int, purpose: str) -> None:
    background.add_task(_email_link, request.app.state.pool, user_id, purpose)


def use_link(conn, token: str, purpose: str) -> int:
    """Marks the token used, once; returns its user."""
    row = conn.execute(
        "UPDATE email_tokens SET used_at = now() WHERE id = %s AND purpose = %s AND used_at IS NULL "
        "AND expires_at > now() RETURNING user_id",
        (sessions.token_hash(token), purpose),
    ).fetchone()
    if row is None:
        raise HTTPException(400, BAD_LINK)
    return row["user_id"]


def _throttle(conn, email: str) -> None:
    row = conn.execute(
        "SELECT count(*) AS failures, min(at) + %s - now() AS wait FROM login_failures "
        "WHERE email = %s AND at > now() - %s",
        (FAILURE_WINDOW, email, FAILURE_WINDOW),
    ).fetchone()
    if row["failures"] >= MAX_FAILURES:
        minutes = max(1, math.ceil(row["wait"].total_seconds() / 60))
        raise HTTPException(429, f"Too many attempts — try again in {minutes} minute{'' if minutes == 1 else 's'}")


def _failed(conn, email: str) -> None:
    conn.execute("DELETE FROM login_failures WHERE at < now() - %s", (FAILURE_WINDOW,))
    conn.execute("INSERT INTO login_failures (email) VALUES (%s)", (email,))


# ---- the routes

@router.post("/signup", status_code=201)
def signup(body: SignupIn, request: Request, background: BackgroundTasks, conn=Depends(get_conn)):
    email, name, slug = normal_email(body.email), check_name(body.name), check_slug(body.workspace)
    if not EMAIL.fullmatch(email):
        raise HTTPException(422, "Enter a valid email address")
    check_password(body.password)
    if not invited(conn, email):
        raise HTTPException(403, NOT_INVITED)
    password_hash = hasher.hash(body.password)
    try:
        with conn.transaction():
            user = conn.execute("SELECT id, email_verified_at FROM users WHERE email = %s FOR UPDATE",
                                (email,)).fetchone()
            if user is None:
                user_id = conn.execute("INSERT INTO users (email, name, password_hash) VALUES (%s, %s, %s) "
                                       "RETURNING id", (email, name, password_hash)).fetchone()["id"]
                create_workspace(conn, user_id, slug)
            elif user["email_verified_at"] is None:
                # signing up again before verifying replaces the pending sign-up, so nobody can hold
                # an invited address by getting there first; the emailed link still works
                user_id = user["id"]
                conn.execute("UPDATE users SET name = %s, password_hash = %s WHERE id = %s",
                             (name, password_hash, user_id))
                conn.execute("UPDATE workspaces SET slug = %s, name = %s WHERE id = "
                             "(SELECT workspace_id FROM memberships WHERE user_id = %s)", (slug, slug, user_id))
            else:
                raise HTTPException(409, EXISTS)
    except psycopg.errors.UniqueViolation as error:   # a race with another sign-up
        raise HTTPException(409, TAKEN if error.diag.table_name == "workspaces" else EXISTS)
    email_link(request, background, user_id, "verify")
    return {"email": email}


@router.post("/verify")
def verify(body: TokenIn, response: Response, conn=Depends(get_conn)):
    with conn.transaction():
        user_id = use_link(conn, body.token, "verify")
        conn.execute("UPDATE users SET email_verified_at = coalesce(email_verified_at, now()) WHERE id = %s",
                     (user_id,))
        sessions.sign_in(conn, response, user_id)
    return me(conn, user_id)


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response, background: BackgroundTasks,
          conn=Depends(get_conn)):
    email = normal_email(body.email)
    _throttle(conn, email)
    user = conn.execute("SELECT id, password_hash, email_verified_at FROM users WHERE email = %s",
                        (email,)).fetchone()
    if not password_ok(user, body.password):
        _failed(conn, email)
        raise HTTPException(401, WRONG)
    conn.execute("DELETE FROM login_failures WHERE email = %s", (email,))
    if user["email_verified_at"] is None:
        email_link(request, background, user["id"], "verify")
        # a Response, not an exception, so the email still goes out
        return JSONResponse({"detail": "Confirm your email first: we've sent you a new link"}, status_code=403)
    if hasher.check_needs_rehash(user["password_hash"]):
        conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (hasher.hash(body.password), user["id"]))
    sessions.sign_in(conn, response, user["id"])
    return me(conn, user["id"])


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, conn=Depends(get_conn)):
    sessions.sign_out(conn, request, response)


@router.post("/password/forgot")
def forgot(body: EmailIn, request: Request, background: BackgroundTasks, conn=Depends(get_conn)):
    user = conn.execute("SELECT id FROM users WHERE email = %s", (normal_email(body.email),)).fetchone()
    if user:
        email_link(request, background, user["id"], "reset")
    return {"detail": FORGOT_SENT}


@router.post("/password/reset")
def reset(body: ResetIn, response: Response, conn=Depends(get_conn)):
    check_password(body.password)
    password_hash = hasher.hash(body.password)
    with conn.transaction():
        user_id = use_link(conn, body.token, "reset")
        # the link came to their inbox, so the email is theirs
        conn.execute("UPDATE users SET password_hash = %s, email_verified_at = coalesce(email_verified_at, now()) "
                     "WHERE id = %s", (password_hash, user_id))
        conn.execute("UPDATE email_tokens SET used_at = now() WHERE user_id = %s AND purpose = 'reset' "
                     "AND used_at IS NULL", (user_id,))
        sessions.end_all(conn, user_id)
        sessions.sign_in(conn, response, user_id)
    return me(conn, user_id)
