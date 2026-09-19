"""
queue.py
--------
The task queue, in the tasks table (docs/PHASE2.md §5).

    enqueue(conn, kind, job_id)   -> task id
    claim(conn, kinds=None)       -> the task, now running, or None
    heartbeat(conn, task_id)
    finish(conn, task_id, error=None)
    requeue_stale(conn)           -> [(task id, "queued" | "failed")]

A train task is claimed only while its job's sandbox host runs fewer train
tasks than its max_running_jobs, and its workspace runs no other: one person's
jobs can't take every place (docs/PHASE2B.md). Plan tasks don't count, since
planning only reads the host. A task that can't run yet is passed over, not
waited on. Claims are serialised with an advisory lock, so two can't both see
room, and FOR UPDATE SKIP LOCKED keeps them apart when there are several workers.
"""

from __future__ import annotations

import psycopg

from worker.db import FILES_KEPT_DAYS, jsonb

CLAIM_LOCK = 72002
STALE_SECONDS = 60
MAX_TRIES = 3


def enqueue(conn: psycopg.Connection, kind: str, job_id: int, payload: dict | None = None) -> int:
    row = conn.execute(
        "INSERT INTO tasks (kind, job_id, payload) VALUES (%s, %s, %s) RETURNING id",
        (kind, job_id, jsonb(payload or {})),
    ).fetchone()
    return row["id"]


def claim(conn: psycopg.Connection, kinds: list | None = None) -> dict | None:
    """The oldest task that may run now; kinds limits which kinds are taken."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (CLAIM_LOCK,))
        return conn.execute(
            """
            UPDATE tasks SET status = 'running', tries = tries + 1, claimed_at = now(), heartbeat_at = now()
            WHERE id = (
                SELECT t.id FROM tasks t
                JOIN jobs j ON j.id = t.job_id
                JOIN sandbox_hosts h ON h.id = j.sandbox_host_id
                WHERE t.status = 'queued'
                  AND (%(kinds)s::text[] IS NULL OR t.kind = ANY(%(kinds)s::text[]))
                  AND (t.kind <> 'train' OR (
                      (SELECT count(*) FROM tasks r JOIN jobs rj ON rj.id = r.job_id
                       WHERE r.status = 'running' AND r.kind = 'train' AND rj.sandbox_host_id = h.id
                      ) < h.max_running_jobs
                      AND NOT EXISTS (
                          SELECT 1 FROM tasks r JOIN jobs rj ON rj.id = r.job_id
                          WHERE r.status = 'running' AND r.kind = 'train' AND rj.workspace_id = j.workspace_id)
                  ))
                ORDER BY t.id
                LIMIT 1
                FOR UPDATE OF t SKIP LOCKED
            )
            RETURNING *
            """,
            {"kinds": kinds},
        ).fetchone()


def heartbeat(conn: psycopg.Connection, task_id: int) -> None:
    conn.execute("UPDATE tasks SET heartbeat_at = now() WHERE id = %s AND status = 'running'", (task_id,))


def finish(conn: psycopg.Connection, task_id: int, error: str | None = None) -> None:
    conn.execute(
        "UPDATE tasks SET status = %s, error = %s WHERE id = %s",
        ("failed" if error else "done", error, task_id),
    )


def requeue_stale(conn: psycopg.Connection) -> list:
    """Put back tasks whose worker stopped heartbeating; after MAX_TRIES, fail them and their job."""
    with conn.transaction():
        rows = conn.execute(
            """
            UPDATE tasks SET
                status = CASE WHEN tries >= %(max)s THEN 'failed' ELSE 'queued' END,
                error = CASE WHEN tries >= %(max)s THEN 'the worker stopped answering ' || tries || ' times'
                        ELSE error END
            WHERE status = 'running' AND heartbeat_at < now() - make_interval(secs => %(stale)s)
            RETURNING id, job_id, status, error
            """,
            {"max": MAX_TRIES, "stale": STALE_SECONDS},
        ).fetchall()
        for row in rows:
            if row["status"] == "failed":
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error = %s, finished_at = now(), "
                    "files_expire_at = now() + make_interval(days => %s) WHERE id = %s",
                    (row["error"], FILES_KEPT_DAYS, row["job_id"]),
                )
    for row in rows:
        print(f"task {row['id']} (job {row['job_id']}): heartbeat stale, {row['status']}", flush=True)
    return [(row["id"], row["status"]) for row in rows]
