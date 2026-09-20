"""
schemas.py
----------
The draft schema, the leakage screen, locking, and the versions behind it
(docs/PHASE4.md §5, §6). Nothing here calls an LLM either.

    GET  · PUT  …/sources/{id}/schema           the draft, with your edits
    POST        …/sources/{id}/schema/analyze   queue the leakage screen; needs a target
    POST        …/sources/{id}/schema/lock      freeze the draft as the next version
    GET         …/sources/{id}/schema/versions  the versions, and ?diff=a,b

A locked schema is never edited. The first edit after a lock copies it into a
new draft at version + 1, so a job that pinned version n keeps the schema it ran
with (§8.6).
"""

from __future__ import annotations

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

import db
from api import sources, tenancy
from api.sessions import current_user, get_conn

router = tenancy.router

TYPES = {"numeric", "categorical", "text", "json", "ordinal", "long", "latitude"}
ROLES = {"feature", "identifier", "ignore", "target"}
# what a person may change about a column; its name and storage dtype are facts about the file (§8.5)
EDITABLE = ("data_type", "role", "description", "available_at_prediction")


class ColumnIn(BaseModel):
    name: str
    data_type: str
    role: str
    description: str = Field(default="", max_length=2000)
    available_at_prediction: bool | None = None


class SchemaIn(BaseModel):
    columns: list[ColumnIn]
    description: str = Field(default="", max_length=2000)


def _source(conn, workspace: dict, source_id: int) -> dict:
    return sources._row(conn, workspace["id"], source_id)


def _schema(conn, source_id: int, status: str | None = None) -> dict | None:
    return conn.execute(
        "SELECT * FROM schemas WHERE source_id = %s AND (%s::text IS NULL OR status = %s) "
        "ORDER BY version DESC LIMIT 1",
        (source_id, status, status),
    ).fetchone()


def _public(row: dict, source: dict) -> dict:
    analysis = row["analysis"] or {}
    return {
        "id": row["id"], "version": row["version"], "status": row["status"],
        "columns": row["columns"]["columns"], "description": row["columns"].get("description", ""),
        "locked_at": row["locked_at"], "created_at": row["created_at"], "updated_at": row["updated_at"],
        "findings": analysis.get("findings", []),
        "analyzed_target": analysis.get("target"),
        # the profile's own numbers per column, so the screen can show null share and distinct counts
        "profile": sources.columns_of(source["profile"]),
        "task": None if not source["task_kind"] else {"kind": source["task_kind"], "status": source["task_status"]},
    }


@router.get("/sources/{source_id}/schema")
def get_schema(source_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    source = _source(conn, workspace, source_id)
    row = _schema(conn, source_id)
    if row is None:
        raise HTTPException(404, "This dataset has no schema yet")
    return _public(row, source)


@router.put("/sources/{source_id}/schema")
def put_schema(source_id: int, body: SchemaIn, workspace: dict = Depends(tenancy.workspace),
               conn=Depends(get_conn)):
    """Your edits. Editing a locked schema starts the next version as a draft (§8.6)."""
    source = _source(conn, workspace, source_id)
    row = _schema(conn, source_id)
    if row is None:
        raise HTTPException(404, "This dataset has no schema yet")

    known = {column["name"] for column in row["columns"]["columns"]}
    sent = {column.name for column in body.columns}
    if sent != known:
        raise HTTPException(422, "The columns don't match the file; reload the screen")
    for column in body.columns:
        if column.data_type not in TYPES:
            raise HTTPException(422, f"{column.name}: {column.data_type} isn't a type we know")
        if column.role not in ROLES:
            raise HTTPException(422, f"{column.name}: {column.role} isn't a role we know")
    targets = [column.name for column in body.columns if column.role == "target"]
    if len(targets) > 1:
        raise HTTPException(422, f"Only one column can be the target; you marked {len(targets)}")

    # keep each column's order and its facts; only the four editable fields move
    by_name = {column.name: column for column in body.columns}
    columns = []
    for old in row["columns"]["columns"]:
        edit = by_name[old["name"]]
        role = edit.role
        columns.append({**old, "data_type": edit.data_type, "role": role, "is_target": role == "target",
                        "description": edit.description, "available_at_prediction": edit.available_at_prediction})
    written = {**row["columns"], "columns": columns, "description": body.description}

    analysis = dict(row["analysis"] or {})
    with conn.transaction():
        if row["status"] == "locked":
            # the locked version stays exactly as the job that used it saw it
            new_id = conn.execute(
                "INSERT INTO schemas (source_id, version, status, columns, analysis) "
                "VALUES (%s, %s, 'draft', %s, %s) RETURNING id",
                (source_id, row["version"] + 1, db.jsonb(written), db.jsonb(analysis)),
            ).fetchone()["id"]
        else:
            new_id = row["id"]
            conn.execute("UPDATE schemas SET columns = %s, analysis = %s, updated_at = now() WHERE id = %s",
                         (db.jsonb(written), db.jsonb(analysis), new_id))
        sources._touch(conn, source_id)
    return _public(conn.execute("SELECT * FROM schemas WHERE id = %s", (new_id,)).fetchone(),
                   _source(conn, workspace, source_id))


@router.post("/sources/{source_id}/schema/analyze", status_code=202)
def analyze(source_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """Queue the leakage screen over the file. Needs a target, and the file still here."""
    source = _source(conn, workspace, source_id)
    if source["status"] != "ready":
        raise HTTPException(409, "This dataset isn't ready to analyse")
    row = _schema(conn, source_id, "draft")
    if row is None:
        raise HTTPException(409, "Edit the draft first; a locked schema is analysed as it was")
    if not any(column["role"] == "target" for column in row["columns"]["columns"]):
        raise HTTPException(422, "Choose the target column first")
    if source["task_kind"]:
        raise HTTPException(409, "Something is already running on this dataset")
    with conn.transaction():
        sources._queue(conn, "analyze", source_id)
        sources._touch(conn, source_id)
    return _public(row, _source(conn, workspace, source_id))


@router.post("/sources/{source_id}/schema/lock")
def lock(source_id: int, workspace: dict = Depends(tenancy.workspace),
         user: dict = Depends(current_user), conn=Depends(get_conn)):
    """Freeze the draft. All it needs is exactly one target.

    The warnings do not gate this. Every one of them is already in front of the
    planners — information_agent.profile() profiles the file again and hands the
    whole profile over — and a leak nobody declared known at prediction time is
    excluded by the run's own policy. A click here would have fed into nothing.
    """
    source = _source(conn, workspace, source_id)
    row = _schema(conn, source_id, "draft")
    if row is None:
        raise HTTPException(409, "There is no draft to lock")
    targets = [column["name"] for column in row["columns"]["columns"] if column["role"] == "target"]
    if len(targets) != 1:
        raise HTTPException(422, "Mark exactly one column as the target before locking")
    with conn.transaction():
        conn.execute("UPDATE schemas SET status = 'locked', locked_at = now(), locked_by = %s, "
                     "updated_at = now() WHERE id = %s", (user["id"], row["id"]))
        sources._touch(conn, source_id)
    return _public(conn.execute("SELECT * FROM schemas WHERE id = %s", (row["id"],)).fetchone(),
                   _source(conn, workspace, source_id))


@router.get("/sources/{source_id}/schema/versions")
def versions(source_id: int, diff: str | None = None, workspace: dict = Depends(tenancy.workspace),
             conn=Depends(get_conn)):
    _source(conn, workspace, source_id)
    rows = conn.execute(
        "SELECT s.id, s.version, s.status, s.columns, s.locked_at, s.created_at, s.updated_at, u.name AS locked_by "
        "FROM schemas s LEFT JOIN users u ON u.id = s.locked_by WHERE s.source_id = %s ORDER BY s.version DESC",
        (source_id,),
    ).fetchall()
    listing = [{key: row[key] for key in ("id", "version", "status", "locked_at", "created_at", "updated_at",
                                          "locked_by")} for row in rows]
    out = {"versions": listing, "diff": None}
    if diff:
        try:
            left, right = (int(part) for part in diff.split(","))
        except ValueError:
            raise HTTPException(422, "diff takes two version numbers, like ?diff=1,2")
        by_version = {row["version"]: row for row in rows}
        if left not in by_version or right not in by_version:
            raise HTTPException(404, "No such version")
        out["diff"] = {"from": left, "to": right,
                       "columns": compare(by_version[left]["columns"], by_version[right]["columns"])}
    return out


def compare(left: dict, right: dict) -> list:
    """Column by column, only what changed — the versions list shows exactly these (§9)."""
    before = {column["name"]: column for column in left["columns"]}
    after = {column["name"]: column for column in right["columns"]}
    out = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old is None or new is None:
            out.append({"name": name, "change": "added" if old is None else "removed", "fields": []})
            continue
        fields = [{"field": field, "from": old.get(field), "to": new.get(field)}
                  for field in EDITABLE if old.get(field) != new.get(field)]
        if fields:
            out.append({"name": name, "change": "changed", "fields": fields})
    return out
