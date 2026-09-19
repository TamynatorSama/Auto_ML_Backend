"""
workspaces.py
-------------
/api/me, creating your workspace, and the workspace itself. One workspace per
person in v1, made at sign-up or on the Welcome screen after a social sign-up.
"""

import re

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api import tenancy
from api.sessions import current_user, get_conn

SLUG = re.compile(r"[a-z][a-z0-9-]{2,31}")
RESERVED = {"api", "app", "auth", "login", "signup", "settings", "admin", "welcome", "reset", "verify",
            "www", "static", "assets", "docs", "help", "status"}
TAKEN = "That workspace address is taken"

router = APIRouter()


class NameIn(BaseModel):
    name: str = Field(max_length=100)


class WorkspaceIn(BaseModel):
    slug: str = Field(max_length=64)


def check_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(422, "Enter a name")
    return name


def check_slug(slug: str) -> str:
    if not SLUG.fullmatch(slug):
        raise HTTPException(422, "A workspace address is 3–32 lowercase letters, digits and hyphens, "
                                 "starting with a letter")
    if slug in RESERVED:
        raise HTTPException(422, "That workspace address is reserved")
    return slug


def create_workspace(conn, user_id: int, slug: str) -> None:
    """In the caller's transaction; a taken slug raises UniqueViolation."""
    workspace_id = conn.execute("INSERT INTO workspaces (slug, name) VALUES (%s, %s) RETURNING id",
                                (slug, slug)).fetchone()["id"]
    conn.execute("INSERT INTO memberships (user_id, workspace_id, role) VALUES (%s, %s, 'owner')",
                 (user_id, workspace_id))


def me(conn, user_id: int) -> dict:
    """What the web app needs after signing in: the user and their workspace (null: go to Welcome)."""
    user = conn.execute("SELECT id, email, name, password_hash IS NOT NULL AS has_password FROM users "
                        "WHERE id = %s", (user_id,)).fetchone()
    workspace = conn.execute(
        "SELECT w.slug, w.name, m.role FROM memberships m JOIN workspaces w ON w.id = m.workspace_id "
        "WHERE m.user_id = %s ORDER BY m.created_at LIMIT 1",
        (user_id,),
    ).fetchone()
    return {"user": user, "workspace": workspace}


@router.get("/api/me")
def get_me(user: dict = Depends(current_user), conn=Depends(get_conn)):
    return me(conn, user["id"])


@router.patch("/api/me")
def rename_me(body: NameIn, user: dict = Depends(current_user), conn=Depends(get_conn)):
    conn.execute("UPDATE users SET name = %s WHERE id = %s", (check_name(body.name), user["id"]))
    return me(conn, user["id"])


@router.post("/api/workspaces", status_code=201)
def create(body: WorkspaceIn, user: dict = Depends(current_user), conn=Depends(get_conn)):
    slug = check_slug(body.slug)
    try:
        with conn.transaction():
            # the user's row, locked: two requests at once can't both make a workspace
            conn.execute("SELECT 1 FROM users WHERE id = %s FOR UPDATE", (user["id"],))
            if conn.execute("SELECT 1 FROM memberships WHERE user_id = %s", (user["id"],)).fetchone():
                raise HTTPException(409, "You already have a workspace")
            create_workspace(conn, user["id"], slug)
    except psycopg.errors.UniqueViolation:
        raise HTTPException(409, TAKEN)
    return me(conn, user["id"])


def _public(workspace: dict) -> dict:
    return {key: workspace[key] for key in ("slug", "name", "role", "created_at")}


@tenancy.router.get("")
def get_workspace(workspace: dict = Depends(tenancy.workspace)):
    return _public(workspace)


@tenancy.router.patch("")
def rename_workspace(body: NameIn, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    name = check_name(body.name)
    conn.execute("UPDATE workspaces SET name = %s WHERE id = %s", (name, workspace["id"]))
    return {**_public(workspace), "name": name}
