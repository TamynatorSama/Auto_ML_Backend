"""
jobs.py
-------
Starting a run and reading it back (docs/PHASE5.md §5, §6).

    POST  …/sources/{id}/jobs   the jobs row and its plan task
    GET   …/jobs · …/jobs/{id}  the workspace's runs; one run with its drafted plan
    PATCH …/jobs/{id}/plan      your edits to the draft, and the run's limits
    POST  …/jobs/{id}/start     approve: store the edits and queue the train task
    POST  …/jobs/{id}/redraft   throw the paused thread away and plan again
    DELETE …/jobs/{id}          the run and its rows; its files go with the next sweep
    POST  …/jobs/{id}/stop · /resume
    GET   …/jobs/{id}/live      ?after={cursor} — the one poll the Training screen makes
    GET   …/jobs/{id}/models/{m}/attempts/{n}/code   the script that attempt ran
    GET   …/jobs/{id}/report    the stored RunReport, unchanged
    GET   …/jobs/{id}/files/{model}/{attempt}/{name}  a file out of the run directory, as bytes

The state changes themselves are db/jobs.py's, shared with worker/cli.py, so a
stop from the browser and a stop from the command line are the same stop. The
API never imports the worker: the plan task is a row db/jobs.py writes.

Edits are stored, never merged: review_plan merges them over its own draft and
validates them by building SplitPlan(**…) and Configs(**…), so a field this file
has never heard of still reaches the pipeline (§5).

A file is served, never opened: bytes out of the run directory, resolved a name
at a time, and nothing on this side ever unpickles a model (§8.8).

The live view is derived, not stored (§8.2): sandboxes running is job_models,
and every gauge is a sum over the ledger, which is indexed by job. Nothing
writes a per-job usage row every few seconds only for the next tick to overwrite.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

import db
from api import tenancy
from api.sessions import current_user, get_conn
from db import jobs as job_state

router = tenancy.router
PROVIDER = "gemini"   # the provider a run needs a key for today (A2)
EVENT_PAGE = 300
# already summed in the usage panel, and 50 of them in a short run: they would bury the log
LEDGER_KINDS = ("llm_usage", "sandbox_usage")
RUNS_ROOT = Path(os.environ.get("AUTOML_RUNS_ROOT") or "runs")


class JobIn(BaseModel):
    name: str | None = Field(default=None, max_length=200)


class PlanIn(BaseModel):
    """Whatever the screen changed. split_plan and config go to the pipeline as they are."""
    split_plan: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    budget_usd: float | None = Field(default=None, gt=0)
    deadline_seconds: int | None = Field(default=None, gt=0)


def _reviewing(row: dict) -> None:
    if row["status"] != "review":
        raise HTTPException(409, "This run isn't waiting for review")


def _save_edits(conn, job_id: int, body: PlanIn) -> None:
    conn.execute(
        """
        UPDATE jobs SET edits = %s,
            budget_usd = coalesce(%s, budget_usd), deadline_seconds = coalesce(%s, deadline_seconds)
        WHERE id = %s
        """,
        (db.jsonb({"split_plan": body.split_plan, "config": body.config}),
         body.budget_usd, body.deadline_seconds, job_id),
    )


def _public(row: dict, plan: bool = False) -> dict:
    out = {
        "id": row["id"], "name": row["name"], "status": row["status"], "error": row["error"],
        "source": {"id": row["source_id"], "name": row["source_name"]},
        "schema_id": row["schema_id"], "budget_usd": row["budget_usd"],
        "deadline_seconds": row["deadline_seconds"], "usage_totals": row["usage_totals"],
        "created_at": row["created_at"], "started_at": row["started_at"], "finished_at": row["finished_at"],
        "files_expire_at": row["files_expire_at"], "files_expired": row["files_expired"],
        "queue_position": row["queue_position"], "sandbox_parallel": row["sandbox_parallel"],
    }
    if plan:
        out["plan"], out["edits"] = row["plan"], row["edits"]
    return out


# how many queued tasks are ahead of this job's own, so "waiting" can say what for
QUEUED = """
LEFT JOIN LATERAL (
    SELECT (SELECT count(*) FROM tasks ahead WHERE ahead.status = 'queued' AND ahead.id < t.id) AS queue_position
    FROM tasks t WHERE t.job_id = j.id AND t.status = 'queued' ORDER BY t.id LIMIT 1
) q ON true
"""
# how many jobs the run's host trains at once, for the Config screen's compute card
COLUMNS = ("j.*, s.name AS source_name, q.queue_position, h.max_running_jobs AS sandbox_parallel")
HOST = "JOIN sandbox_hosts h ON h.id = j.sandbox_host_id"


def _row(conn, workspace_id: int, job_id: int) -> dict:
    row = conn.execute(
        f"SELECT {COLUMNS} FROM jobs j JOIN sources s ON s.id = j.source_id {HOST} {QUEUED} "
        "WHERE j.id = %s AND j.workspace_id = %s",
        (job_id, workspace_id),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Not found")
    return row


@router.post("/sources/{source_id}/jobs", status_code=201)
def create(source_id: int, body: JobIn, workspace: dict = Depends(tenancy.workspace),
           user: dict = Depends(current_user), conn=Depends(get_conn)):
    """A locked schema, a verified key and no unfinished run on this dataset (D4)."""
    source = conn.execute(
        """
        SELECT s.id, s.name, s.status, sc.id AS schema_id, sc.status AS schema_status
        FROM sources s
        LEFT JOIN LATERAL (
            SELECT id, status FROM schemas WHERE source_id = s.id ORDER BY version DESC LIMIT 1
        ) sc ON true
        WHERE s.id = %s AND s.workspace_id = %s
        """,
        (source_id, workspace["id"]),
    ).fetchone()
    if source is None:
        raise HTTPException(404, "Not found")
    if source["status"] != "ready":
        raise HTTPException(409, "This dataset isn't ready yet")
    if source["schema_status"] != "locked":
        raise HTTPException(409, "Lock the schema before starting a run")
    if not conn.execute("SELECT 1 FROM provider_keys WHERE workspace_id = %s AND provider = %s",
                        (workspace["id"], PROVIDER)).fetchone():
        raise HTTPException(409, "Add a model key under Settings before starting a run")

    try:
        job_id = job_state.create(conn, workspace["id"], source_id, source["schema_id"],
                                  (body.name or source["name"]).strip() or source["name"], started_by=user["id"])
    except job_state.JobError as error:
        raise HTTPException(409, str(error))
    return _public(_row(conn, workspace["id"], job_id))


@router.get("/jobs")
def listing(workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM jobs j JOIN sources s ON s.id = j.source_id {HOST} {QUEUED} "
        "WHERE j.workspace_id = %s ORDER BY j.id DESC",
        (workspace["id"],),
    ).fetchall()
    return {"jobs": [_public(row) for row in rows]}


@router.get("/jobs/{job_id}")
def one(job_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    return _public(_row(conn, workspace["id"], job_id), plan=True)


@router.patch("/jobs/{job_id}/plan")
def edit(job_id: int, body: PlanIn, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """Kept so the screen survives a reload; nothing runs until start."""
    _reviewing(_row(conn, workspace["id"], job_id))
    _save_edits(conn, job_id, body)
    return _public(_row(conn, workspace["id"], job_id), plan=True)


@router.post("/jobs/{job_id}/start")
def start(job_id: int, body: PlanIn, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """Approve: the graph resumes from its interrupt with these edits and training begins."""
    row = _row(conn, workspace["id"], job_id)
    _reviewing(row)
    try:
        with conn.transaction():
            _save_edits(conn, job_id, body)
            job_state.approve(conn, job_id, {"split_plan": body.split_plan, "config": body.config})
    except job_state.JobError as error:
        raise HTTPException(409, str(error))
    return _public(_row(conn, workspace["id"], job_id), plan=True)


@router.delete("/jobs/{job_id}", status_code=204)
def remove(job_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """A run that has ended, and everything recorded against it — including its usage."""
    _row(conn, workspace["id"], job_id)
    try:
        job_state.remove(conn, job_id)
    except job_state.JobError as error:
        raise HTTPException(409, str(error))


@router.post("/jobs/{job_id}/stop")
def stop(job_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """A run whose task is still queued stops at once; a training one at its next check."""
    _row(conn, workspace["id"], job_id)
    try:
        job_state.stop(conn, job_id)
    except job_state.JobError as error:
        raise HTTPException(409, str(error))
    return _public(_row(conn, workspace["id"], job_id), plan=True)


@router.post("/jobs/{job_id}/resume")
def resume(job_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """Carry a stopped or failed run on from its checkpoint."""
    _row(conn, workspace["id"], job_id)
    try:
        job_state.resume(conn, job_id)
    except job_state.JobError as error:
        raise HTTPException(409, str(error))
    return _public(_row(conn, workspace["id"], job_id), plan=True)


@router.get("/jobs/{job_id}/live")
def live(job_id: int, after: int = 0, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """One poll: the job, its gauges, its models, its attempts, and the events since `after`."""
    row = _row(conn, workspace["id"], job_id)

    events = conn.execute(
        "SELECT id, at, kind, model, data FROM job_events "
        "WHERE job_id = %s AND id > %s AND NOT (kind = ANY(%s)) ORDER BY id LIMIT %s",
        (job_id, after, list(LEDGER_KINDS), EVENT_PAGE),
    ).fetchall()
    # the cursor moves past the ledger events too, or they would be re-read for ever;
    # but not past an event this page had no room for
    newest = conn.execute("SELECT coalesce(max(id), 0) AS id FROM job_events WHERE job_id = %s",
                          (job_id,)).fetchone()["id"]
    cursor = events[-1]["id"] if len(events) == EVENT_PAGE else max(newest, after)

    return {
        "job": _public(row),
        "usage": _usage(conn, job_id),
        "models": conn.execute(
            "SELECT model, status, generation, best_attempt, best_cv, test_scores "
            "FROM job_models WHERE job_id = %s ORDER BY model", (job_id,)).fetchall(),
        "attempts": conn.execute(
            "SELECT id, model, attempt, generation, kind, status, cv_scores, wall_seconds, changes, at "
            "FROM job_attempts WHERE job_id = %s ORDER BY id", (job_id,)).fetchall(),
        "events": events,
        "cursor": cursor,
    }


def _usage(conn, job_id: int) -> dict:
    """Derived, never stored (§8.2): the ledger is indexed by job, and job_models says what is running."""
    meters = {row["meter"]: float(row["total"]) for row in conn.execute(
        "SELECT meter, sum(quantity) AS total FROM usage_records WHERE job_id = %s GROUP BY meter", (job_id,))}
    running = conn.execute(
        "SELECT count(*) AS n FROM job_models WHERE job_id = %s AND status = 'running'", (job_id,)
    ).fetchone()["n"]
    return {"running": running, "meters": meters}


@router.get("/jobs/{job_id}/models/{model}/attempts/{attempt}/code", response_class=PlainTextResponse)
def code(job_id: int, model: str, attempt: int, workspace: dict = Depends(tenancy.workspace),
         conn=Depends(get_conn)):
    """The script that attempt ran, as text. Read, never executed and never imported."""
    row = _row(conn, workspace["id"], job_id)
    path = in_run_dir(row, model, f"attempt_{attempt}", "candidate.py")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise HTTPException(404, "That script isn't here any more")


def in_run_dir(row: dict, *parts: str) -> Path:
    """A path inside this job's run directory, or a 404. Nothing else is ever served (§8.8).

    Each part must be one plain name: no separators, no .., no drive or root, so a
    name out of the database or the address cannot climb out of the directory. The
    result is checked against the resolved run directory as well, which catches a
    symlink pointing elsewhere.

    A trailing dot or space, and NTFS's name:stream, are refused too. Windows strips
    them while resolving, so "model.joblib." and "model.joblib::$DATA" both reached the
    same file through a name that is not the file's name. Nothing escaped the run
    directory — the resolved path is checked against it either way — but the rule here
    says one plain name, and now it means it on both platforms.
    """
    if row["files_expired"] or not row["run_dir"]:
        raise HTTPException(404, "This run's files have been deleted")
    for part in parts:
        if (not part or part in (".", "..") or Path(part).name != part
                or part != part.rstrip(". ") or ":" in part):
            raise HTTPException(404, "Not found")
    run_dir = Path(row["run_dir"]).resolve()
    root = RUNS_ROOT.resolve()
    if not run_dir.is_relative_to(root) or run_dir == root:
        raise HTTPException(404, "Not found")
    path = run_dir.joinpath(*parts).resolve()
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise HTTPException(404, "Not found")
    return path


@router.get("/jobs/{job_id}/report")
def report(job_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """The RunReport the pipeline wrote, as it wrote it."""
    row = _row(conn, workspace["id"], job_id)
    if not row["report"]:
        raise HTTPException(409, "This run has no report yet")
    return row["report"]


@router.get("/jobs/{job_id}/files/{model}/{attempt}/{name}")
def artifact(job_id: int, model: str, attempt: int, name: str,
             workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """A file the run produced, streamed as bytes.

    The name is resolved against this job's run directory and nothing else. It is
    never loaded, parsed or unpickled here — a model file is a download, and
    joblib.load on it would run whatever it contains (§8.8).
    """
    row = _row(conn, workspace["id"], job_id)
    path = in_run_dir(row, model, f"attempt_{attempt}", name)
    return FileResponse(path, media_type="application/octet-stream", filename=f"{model}-attempt-{attempt}-{name}")


@router.post("/jobs/{job_id}/redraft")
def redraft(job_id: int, workspace: dict = Depends(tenancy.workspace), conn=Depends(get_conn)):
    """Plan again from nothing, for a draft not worth editing."""
    _row(conn, workspace["id"], job_id)
    try:
        job_state.redraft(conn, job_id)
    except job_state.JobError as error:
        raise HTTPException(409, str(error))
    return _public(_row(conn, workspace["id"], job_id), plan=True)
