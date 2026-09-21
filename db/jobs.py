"""
jobs.py
-------
A job's state changes, in plain SQL over jobs and tasks (docs/PHASE5.md §8.1).
The API and worker/cli.py both call these, so the browser and the command line
can never disagree about what a stop or a resume means. They live in db/
because the API may not import the worker.

    host_for(conn, workspace_id)    the workspace's own sandbox host, else the platform's
    create(conn, …)                 the jobs row and its plan task
    approve(conn, job_id, edits)    review -> queued, and a train task
    redraft(conn, job_id)           throw the paused thread away and plan again
    stop(conn, job_id)              -> "stopped" | "stopping"
    resume(conn, job_id)            -> "planning" | "queued"
    remove(conn, job_id)            the run's rows; the worker's sweep takes its files
    limits(conn, job_id)            the cap, the seconds left, and what the ledger says is spent

Each raises JobError when the job is not in a state that allows it: the API
turns that into a 409, the command line into a message.
"""

from __future__ import annotations

from db import jsonb

# a job that hasn't ended: only one of these per source at a time (D4)
UNFINISHED = ("planning", "review", "queued", "running", "stopping")
DEFAULT_BUDGET_USD = 5.0
DEFAULT_DEADLINE_SECONDS = 4 * 3600


class JobError(Exception):
    """The job can't do that from where it is."""


def enqueue(conn, kind: str, job_id: int, payload: dict | None = None) -> int:
    """The same task row worker/queue.py writes, without importing the worker."""
    return conn.execute(
        "INSERT INTO tasks (kind, job_id, payload) VALUES (%s, %s, %s) RETURNING id",
        (kind, job_id, jsonb(payload or {})),
    ).fetchone()["id"]


def host_for(conn, workspace_id: int) -> int:
    """The workspace's own host if it has one, else the platform's."""
    row = conn.execute(
        """
        SELECT id FROM sandbox_hosts
        WHERE (owner_workspace_id = %s OR owner_workspace_id IS NULL) AND status <> 'disabled'
        ORDER BY owner_workspace_id IS NULL, id LIMIT 1
        """,
        (workspace_id,),
    ).fetchone()
    if row is None:
        raise JobError("no sandbox host is registered")
    return row["id"]


def create(conn, workspace_id: int, source_id: int, schema_id: int, name: str,
           budget_usd: float | None = DEFAULT_BUDGET_USD,
           deadline_seconds: int | None = DEFAULT_DEADLINE_SECONDS,
           started_by: int | None = None) -> int:
    """The job and its plan task. schema_id is pinned, so a later schema edit can't change this run."""
    with conn.transaction():
        running = conn.execute(
            "SELECT id FROM jobs WHERE source_id = %s AND status = ANY(%s) LIMIT 1",
            (source_id, list(UNFINISHED)),
        ).fetchone()
        if running:
            raise JobError(f"run {running['id']} on this dataset hasn't finished yet")
        job_id = conn.execute(
            """
            INSERT INTO jobs (workspace_id, source_id, schema_id, sandbox_host_id, name,
                              budget_usd, deadline_seconds, started_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (workspace_id, source_id, schema_id, host_for(conn, workspace_id), name,
             budget_usd, deadline_seconds, started_by),
        ).fetchone()["id"]
        enqueue(conn, "plan", job_id)
    return job_id


def approve(conn, job_id: int, edits: dict | None = None) -> None:
    """The edits are stored, not merged: review_plan itself merges them over its draft (§5)."""
    with conn.transaction():
        updated = conn.execute(
            "UPDATE jobs SET status = 'queued', edits = %s WHERE id = %s AND status = 'review' RETURNING id",
            (jsonb(edits or {}), job_id),
        ).fetchone()
        if updated is None:
            raise JobError(f"job {job_id} is not waiting for review")
        enqueue(conn, "train", job_id)


def redraft(conn, job_id: int) -> None:
    """Plan again from nothing, for a draft not worth editing. Resume is the other choice.

    The plan task carries restart, and the worker throws the graph's thread away
    before it starts: LangGraph's checkpoint tables belong to the worker, which
    creates them, and the API must not reach into them.
    """
    with conn.transaction():
        job = conn.execute("SELECT status FROM jobs WHERE id = %s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["status"] not in ("review", "stopped", "failed"):
            raise JobError(f"job {job_id} has no plan to draft again")
        conn.execute(
            "UPDATE jobs SET status = 'planning', plan = NULL, edits = NULL, error = NULL, run_dir = NULL, "
            "cancel_requested = false, finished_at = NULL, files_expire_at = NULL WHERE id = %s",
            (job_id,),
        )
        enqueue(conn, "plan", job_id, {"restart": True})


def stop(conn, job_id: int) -> str:
    """A job whose task is still queued stops at once; a training one at its next check, between attempts.

    Only training checks for a stop (code_gen_eval calls hooks.check_stop), so a
    plan already under way is left to finish rather than marked stopping forever.
    """
    with conn.transaction():
        job = conn.execute("SELECT status FROM jobs WHERE id = %s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["status"] not in UNFINISHED:
            raise JobError(f"job {job_id} has already finished")
        cancelled = conn.execute(
            "UPDATE tasks SET status = 'cancelled' WHERE job_id = %s AND status = 'queued' RETURNING id", (job_id,)
        ).fetchone()
        if cancelled is None and job["status"] != "running":
            raise JobError("this run is still being planned; stop it once the plan is ready")
        status = "stopped" if cancelled else "stopping"
        conn.execute(
            "UPDATE jobs SET status = %s, cancel_requested = %s WHERE id = %s", (status, not cancelled, job_id)
        )
    return status


def resume(conn, job_id: int) -> str:
    """Carry a stopped or failed job on from where its thread is: plan again if it never got a plan."""
    with conn.transaction():
        job = conn.execute("SELECT status, plan, files_expired FROM jobs WHERE id = %s FOR UPDATE",
                           (job_id,)).fetchone()
        if job is None or job["status"] not in ("stopped", "failed"):
            raise JobError(f"job {job_id} is not stopped or failed")
        if job["files_expired"]:
            raise JobError(f"job {job_id}'s files have expired")
        kind = "plan" if job["plan"] is None else "train"
        status = "planning" if kind == "plan" else "queued"
        conn.execute(
            "UPDATE jobs SET status = %s, cancel_requested = false, error = NULL, finished_at = NULL, "
            "files_expire_at = NULL WHERE id = %s",
            (status, job_id),
        )
        enqueue(conn, kind, job_id)
    return status


# every table that points at a job, in the order they have to go. usage_records is
# not among them: the ledger outlives the run (see remove)
BELONGS_TO_JOB = ("job_events", "job_attempts", "job_models", "tasks")


def remove(conn, job_id: int) -> None:
    """Delete a finished run and the events, attempts and scores recorded against it.

    **The ledger stays.** Its rows lose their job_id and keep their workspace, so
    what a run spent survives the run being deleted: it is the billing record (D10)
    and the only account of how much of a provider quota a day has used. Deleting
    it took that away the first time someone needed it.

    Only rows are touched here: the graph's thread and the run directory belong to
    the worker, which collects them on its next sweep (worker/sweep.py). A job with
    a task still queued or running is refused — stop it first, or the worker would
    carry on writing events for a job that no longer exists.
    """
    with conn.transaction():
        job = conn.execute("SELECT status, run_dir FROM jobs WHERE id = %s FOR UPDATE", (job_id,)).fetchone()
        if job is None:
            raise JobError(f"no job {job_id}")
        if job["status"] in ("planning", "queued", "running", "stopping"):
            raise JobError("this run is still going; stop it before deleting it")
        # what the worker has to collect, written down rather than worked out later
        conn.execute("INSERT INTO deleted_runs (job_id, run_dir) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                     (job_id, job["run_dir"]))
        conn.execute("UPDATE usage_records SET job_id = NULL WHERE job_id = %s", (job_id,))
        for table in BELONGS_TO_JOB:
            conn.execute(f"DELETE FROM {table} WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = %s", (job_id,))


def limits(conn, job_id: int) -> dict:
    """budget_usd · seconds_left · spent_usd.

    The spend comes from the ledger, never from a total held in memory: a worker
    that died and requeued the task must not hand the job a fresh budget (§8.3).
    The deadline is wall-clock from started_at, so a restart doesn't extend it.
    """
    return conn.execute(
        """
        SELECT j.budget_usd,
               CASE WHEN j.deadline_seconds IS NULL THEN NULL ELSE
                    extract(epoch FROM coalesce(j.started_at, now())
                            + make_interval(secs => j.deadline_seconds) - now()) END AS seconds_left,
               (SELECT coalesce(sum(quantity), 0) FROM usage_records
                WHERE job_id = j.id AND meter = 'cost_usd') AS spent_usd
        FROM jobs j WHERE j.id = %s
        """,
        (job_id,),
    ).fetchone()
