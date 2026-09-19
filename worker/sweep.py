"""
sweep.py
--------
The seven-day sweep (A4): a finished job's run directory and checkpoints are
deleted once files_expire_at passes. The job row, its report, events and usage
stay. Sources are left alone: in Phase 2 they point at files the platform
doesn't own (the try-out CSVs).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from worker import jobs


def sweep(pool, saver) -> list:
    """Expire what is due; returns the job ids."""
    with pool.connection() as conn:
        due = conn.execute(
            "SELECT id, run_dir FROM jobs WHERE files_expire_at < now() AND NOT files_expired ORDER BY id"
        ).fetchall()
    root = jobs.RUNS_ROOT.resolve()
    for job in due:
        run_dir = Path(job["run_dir"] or "").resolve()
        # the path comes from the database: never delete outside the runs folder
        if job["run_dir"] and run_dir.is_relative_to(root) and run_dir != root:
            shutil.rmtree(run_dir, ignore_errors=True)
        saver.delete_thread(str(job["id"]))
        with pool.connection() as conn:
            conn.execute("UPDATE jobs SET files_expired = true WHERE id = %s", (job["id"],))
        print(f"job {job['id']}: files expired", flush=True)
    return [job["id"] for job in due]
