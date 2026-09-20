"""
sources.py
----------
Uploading a CSV and everything the Data and Preview screens read
(docs/PHASE4.md §5, §6). Nothing here calls an LLM.

    POST   …/sources             a row, and the address to upload to
    PUT    …/sources/{id}/file   the CSV, streamed and capped at 100 MB
    POST   …/sources/sample      the housing-prices sample, schema filled in
    GET    …/sources             · …/sources/{id}
    GET    …/sources/{id}/rows   a page of rows, read out of the CSV by DuckDB
    DELETE …/sources/{id}

The API never imports the worker: it queues a task by writing the row itself,
the same row worker/queue.py writes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

import db
from api import files, tenancy
from api.sessions import current_user, get_conn

PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
SAMPLE_PATH = Path("data/train.csv")
# unfinished: the source's file is still being written or read
BUSY = ("uploading", "checking", "profiling")

router = tenancy.router


class SourceIn(BaseModel):
    name: str = Field(max_length=200)


def _check_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(422, "Give the dataset a name")
    return name


# The newest schema, with the two counts the Data card shows. Reading the whole columns
# array here keeps the card honest without the screen fetching the schema itself.
SCHEMA = """
LEFT JOIN LATERAL (
    SELECT sc.id, sc.version, sc.status,
           (SELECT count(*) FROM jsonb_array_elements(sc.columns -> 'columns') c
            WHERE coalesce(c ->> 'description', '') <> '') AS described,
           (SELECT c ->> 'name' FROM jsonb_array_elements(sc.columns -> 'columns') c
            WHERE c ->> 'role' = 'target' LIMIT 1) AS target
    FROM schemas sc WHERE sc.source_id = s.id ORDER BY sc.version DESC LIMIT 1
) sc ON true
"""
TASK = """
LEFT JOIN LATERAL (
    SELECT kind, status FROM tasks
    WHERE source_id = s.id AND status IN ('queued', 'running') ORDER BY id DESC LIMIT 1
) t ON true
"""
PICKED = ("sc.id AS schema_id, sc.version AS schema_version, sc.status AS schema_status, "
          "sc.described, sc.target, t.kind AS task_kind, t.status AS task_status")


def _row(conn, workspace_id: int, source_id: int) -> dict:
    """One source, with its newest schema and whatever task is outstanding."""
    row = conn.execute(
        f"SELECT s.*, {PICKED} FROM sources s {SCHEMA} {TASK} "
        "WHERE s.id = %s AND s.workspace_id = %s",
        (source_id, workspace_id),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Not found")
    return row


def _public(row: dict, profile: bool = False) -> dict:
    out = {
        "id": row["id"], "name": row["name"], "original_name": row["original_name"],
        "status": row["status"], "rows": row["rows"], "columns": row["columns"], "bytes": row["bytes"],
        "error": row["error"], "created_at": row["created_at"], "files_expire_at": row["files_expire_at"],
        "schema": None if row.get("schema_id") is None else {
            "id": row["schema_id"], "version": row["schema_version"], "status": row["schema_status"],
            "described": row["described"], "target": row["target"],
        },
        "task": None if not row.get("task_kind") else {"kind": row["task_kind"], "status": row["task_status"]},
    }
    if profile:
        out["profile"] = None if not row["profile"] else {**row["profile"],
                                                          "columns": columns_of(row["profile"])}
    return out


def columns_of(profile: dict | None) -> dict:
    """The profile's per-column entries, with the target among them.

    Once analyze has profiled against a target, profile_dataset reports that
    column under "target" instead of in "columns"; the screens want one map.
    """
    profile = profile or {}
    columns = dict(profile.get("columns") or {})
    target = profile.get("target")
    if target and target.get("name"):
        columns.setdefault(target["name"], target)
    return columns


def _queue(conn, kind: str, source_id: int) -> None:
    """The same task row worker/queue.py writes, without importing the worker."""
    conn.execute("INSERT INTO tasks (kind, source_id) VALUES (%s, %s)", (kind, source_id))


def _touch(conn, source_id: int) -> None:
    """Seven days from when the source was last used, not from upload (A4, §8.7)."""
    conn.execute(
        "UPDATE sources SET last_used_at = now(), files_expire_at = now() + make_interval(days => %s) "
        "WHERE id = %s",
        (db.FILES_KEPT_DAYS, source_id),
    )


@router.post("/sources", status_code=201)
def create(body: SourceIn, workspace: dict = Depends(tenancy.workspace),
           user: dict = Depends(current_user), conn=Depends(get_conn)):
    """A row to upload into. The file itself goes to the returned address."""
    name = _check_name(body.name)
    source_id = conn.execute(
        "INSERT INTO sources (workspace_id, name, original_name, path, status, created_by) "
        "VALUES (%s, %s, %s, '', 'uploading', %s) RETURNING id",
        (workspace["id"], name, name, user["id"]),
    ).fetchone()["id"]
    return {**_public(_row(conn, workspace["id"], source_id)),
            "upload_url": f"/api/w/{workspace['slug']}/sources/{source_id}/file"}


@router.put("/sources/{source_id}/file")
async def upload(source_id: int, request: Request, workspace: dict = Depends(tenancy.workspace),
                 conn=Depends(get_conn)):
    """The CSV itself: streamed to disk, refused the moment it passes 100 MB (D3)."""
    source = _row(conn, workspace["id"], source_id)
    if source["status"] != "uploading":
        raise HTTPException(409, "This dataset already has a file")

    path = files.path_for(workspace["id"], source_id)
    try:
        size = await files.write(path, request.stream())
    except files.TooLarge:
        # nothing left behind: no file, no row (§9)
        conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        raise HTTPException(413, f"A dataset can be at most {files.MAX_BYTES // (1024 * 1024)} MB")
    if not size:
        files.remove(path)
        conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
        raise HTTPException(422, "The file is empty")

    with conn.transaction():
        conn.execute("UPDATE sources SET path = %s, bytes = %s, status = 'checking' WHERE id = %s",
                     (str(path), size, source_id))
        _touch(conn, source_id)
        _queue(conn, "ingest", source_id)
    return _public(_row(conn, workspace["id"], source_id))


@router.post("/sources/sample", status_code=201)
def load_sample(workspace: dict = Depends(tenancy.workspace), user: dict = Depends(current_user),
                conn=Depends(get_conn)):
    """The housing-prices dataset, with the descriptions already written (A8, §8.4)."""
    if not SAMPLE_PATH.exists():
        raise HTTPException(503, "The sample dataset isn't installed on this server")
    with conn.transaction():
        source_id = conn.execute(
            "INSERT INTO sources (workspace_id, name, original_name, path, status, created_by) "
            "VALUES (%s, %s, %s, '', 'uploading', %s) RETURNING id",
            (workspace["id"], SAMPLE["name"], SAMPLE_PATH.name, user["id"]),
        ).fetchone()["id"]
        path = files.path_for(workspace["id"], source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLE_PATH, path)
        conn.execute("UPDATE sources SET path = %s, bytes = %s, status = 'checking' WHERE id = %s",
                     (str(path), path.stat().st_size, source_id))
        # the draft goes in first; profiling fills in the rest and keeps these descriptions
        conn.execute("INSERT INTO schemas (source_id, version, status, columns) VALUES (%s, 1, 'draft', %s)",
                     (source_id, db.jsonb(SAMPLE)))
        _touch(conn, source_id)
        _queue(conn, "ingest", source_id)
    return _public(_row(conn, workspace["id"], source_id))


@router.get("/sources")
def listing(workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    # every column but the profile, which is large and only the Preview screen reads it
    rows = conn.execute(
        f"""
        SELECT s.id, s.name, s.original_name, s.status, s.rows, s.columns, s.bytes, s.error,
               s.created_at, s.files_expire_at, {PICKED}
        FROM sources s {SCHEMA} {TASK}
        WHERE s.workspace_id = %s ORDER BY s.id DESC
        """,
        (workspace["id"],),
    ).fetchall()
    return {"sources": [_public(row) for row in rows]}


@router.get("/sources/{source_id}")
def one(source_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    return _public(_row(conn, workspace["id"], source_id), profile=True)


@router.get("/sources/{source_id}/rows")
def rows(source_id: int, page: int = 1, size: int = PAGE_SIZE,
         workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """A page of the CSV itself. DuckDB reads only that page, whatever the file's size (G7)."""
    source = _row(conn, workspace["id"], source_id)
    if source["status"] in ("uploading", "expired") or not source["path"]:
        raise HTTPException(409, "This dataset's file isn't here")
    page, size = max(page, 1), min(max(size, 1), MAX_PAGE_SIZE)
    columns, values = read_page(source["path"], (page - 1) * size, size)
    return {"page": page, "size": size, "total": source["rows"], "columns": columns, "rows": values}


def read_page(path: str, offset: int, limit: int) -> tuple[list, list]:
    """(column names, rows as lists of strings). Everything is read as text: this is for looking at."""
    import duckdb

    with duckdb.connect() as connection:
        result = connection.execute(
            "SELECT * FROM read_csv(?, header = true, all_varchar = true) LIMIT ? OFFSET ?",
            [str(path), limit, offset],
        )
        return [column[0] for column in result.description], [list(row) for row in result.fetchall()]


@router.delete("/sources/{source_id}", status_code=204)
def delete(source_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    source = _row(conn, workspace["id"], source_id)
    if source["status"] in BUSY or source["task_kind"]:
        raise HTTPException(409, "This dataset is still being read; try again once it's ready")
    if conn.execute("SELECT 1 FROM jobs WHERE source_id = %s", (source_id,)).fetchone():
        raise HTTPException(409, "This dataset has runs; delete those first")
    with conn.transaction():
        conn.execute("DELETE FROM schemas WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM tasks WHERE source_id = %s", (source_id,))
        conn.execute("DELETE FROM sources WHERE id = %s", (source_id,))
    files.remove(source["path"])


# The shipped sample's schema, as a DataSchema. The file's LONGITUDE column holds
# latitudes and its LATITUDE column holds longitudes; the descriptions say so
# rather than the CSV being rewritten, so data/train.csv stays as the pipeline's
# own tests know it (§8.4).
SAMPLE = {
    "name": "Indian housing prices",
    "description": "Property listings across India, with the asking price in lakh rupees",
    "need_real_world_info": False,
    "columns": [
        {"name": "POSTED_BY", "data_type": "categorical", "role": "feature",
         "description": "Who posted the listing: Owner, Dealer or Builder"},
        {"name": "UNDER_CONSTRUCTION", "data_type": "categorical", "role": "feature",
         "description": "Whether the property is under construction (0 = no, 1 = yes)"},
        {"name": "RERA", "data_type": "categorical", "role": "feature",
         "description": "Whether the property is registered under RERA (0 = no, 1 = yes)"},
        {"name": "BHK_NO.", "data_type": "numeric", "role": "feature",
         "description": "Number of bedrooms"},
        {"name": "BHK_OR_RK", "data_type": "categorical", "role": "feature",
         "description": "Layout type: BHK (bedroom, hall, kitchen) or RK (room, kitchen)"},
        {"name": "SQUARE_FT", "data_type": "numeric", "role": "feature",
         "description": "Floor area in square feet"},
        {"name": "READY_TO_MOVE", "data_type": "categorical", "role": "feature",
         "description": "Whether the property is ready to move into (0 = no, 1 = yes)"},
        {"name": "RESALE", "data_type": "categorical", "role": "feature",
         "description": "Whether this is a resale listing (0 = no, 1 = yes)"},
        {"name": "ADDRESS", "data_type": "text", "role": "feature",
         "description": "Locality and city"},
        {"name": "LONGITUDE", "data_type": "latitude", "role": "feature",
         "description": "Latitude of the property. The column is named LONGITUDE but holds a latitude"},
        {"name": "LATITUDE", "data_type": "long", "role": "feature",
         "description": "Longitude of the property. The column is named LATITUDE but holds a longitude"},
        {"name": "TARGET(PRICE_IN_LACS)", "data_type": "numeric", "role": "target", "is_target": True,
         "description": "Asking price in lakh rupees"},
    ],
}
