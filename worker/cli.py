"""
cli.py
------
What the Config screen will do, until the web app has it (Phases 4 and 5):

    python -m worker.cli add-host URL [--workspace SLUG] [--token-env NAME] [--max-running-jobs N]
    python -m worker.cli set-host HOST --max-running-jobs N   how many jobs train on it at once
    python -m worker.cli enqueue SLUG [--workspace SLUG]      a try_out_runner.py dataset
    python -m worker.cli approve JOB [--edit JSON|FILE]       e.g. --edit '{"config": {"eval_matrics": ["f1"]}}'
    python -m worker.cli stop JOB
    python -m worker.cli resume JOB
    python -m worker.cli status [JOB]
    python -m worker.cli allow EMAIL...                       invite to the beta: they can sign up (A5)
    python -m worker.cli disallow EMAIL...

A host without --workspace is the platform's own; with one, it runs only that
workspace's jobs (D11). The token is read from --token-env, or asked for.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import db
from models import DataSchema
from worker import crypto, queue

DEFAULT_WORKSPACE = "default"


def _json_arg(text: str) -> dict:
    # a file is easier than JSON quoting in Windows PowerShell
    return json.loads(Path(text).read_text(encoding="utf-8") if Path(text).is_file() else text)


# ---- building blocks (the tests use them too)

def workspace(conn, slug: str) -> int:
    conn.execute("INSERT INTO workspaces (slug, name) VALUES (%s, %s) ON CONFLICT (slug) DO NOTHING", (slug, slug))
    return conn.execute("SELECT id FROM workspaces WHERE slug = %s", (slug,)).fetchone()["id"]


def add_host(conn, url: str, token: str, key: str, workspace_id: int | None = None,
             max_running_jobs: int = 1, runtime_hash: str | None = None) -> int:
    return conn.execute(
        """
        INSERT INTO sandbox_hosts (owner_workspace_id, url, token_ciphertext, token_last4, runtime_hash,
                                   max_running_jobs, last_seen_at)
        VALUES (%s, %s, %s, %s, %s, %s, now()) RETURNING id
        """,
        (workspace_id, url, crypto.encrypt(key, token), crypto.last4(token), runtime_hash, max_running_jobs),
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
        raise SystemExit("no sandbox host: add one with `python -m worker.cli add-host URL`")
    return row["id"]


def create_job(conn, workspace_id: int, name: str, data_path: str, schema: DataSchema) -> int:
    """A source, its schema locked as version 1, the job, and its plan task."""
    path = Path(data_path)
    with conn.transaction():
        with path.open("rb") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
        # the same row the API writes, minus files_expire_at: a try-out CSV is not ours to delete (§8.8)
        source_id = conn.execute(
            "INSERT INTO sources (workspace_id, name, original_name, path, rows, columns, bytes, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, 'ready') RETURNING id",
            (workspace_id, schema.name, path.name, str(path), rows, len(schema.columns), path.stat().st_size),
        ).fetchone()["id"]
        schema_id = conn.execute(
            "INSERT INTO schemas (source_id, version, status, columns, locked_at) "
            "VALUES (%s, 1, 'locked', %s, now()) RETURNING id",
            (source_id, db.jsonb(schema.model_dump(mode="json"))),
        ).fetchone()["id"]
        job_id = conn.execute(
            "INSERT INTO jobs (workspace_id, source_id, schema_id, sandbox_host_id, name) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (workspace_id, source_id, schema_id, host_for(conn, workspace_id), name),
        ).fetchone()["id"]
        queue.enqueue(conn, "plan", job_id)
    return job_id


def allow(conn, emails: list) -> None:
    for email in emails:
        conn.execute("INSERT INTO signup_allowlist (email) VALUES (%s) ON CONFLICT DO NOTHING",
                     (email.strip().lower(),))


def disallow(conn, emails: list) -> None:
    # stops new sign-ups only; an account already made stays
    conn.execute("DELETE FROM signup_allowlist WHERE email = ANY(%s)", ([email.strip().lower() for email in emails],))


def approve(conn, job_id: int, edits: dict | None = None) -> None:
    with conn.transaction():
        updated = conn.execute(
            "UPDATE jobs SET status = 'queued', edits = %s WHERE id = %s AND status = 'review' RETURNING id",
            (db.jsonb(edits or {}), job_id),
        ).fetchone()
        if updated is None:
            raise SystemExit(f"job {job_id} is not waiting for review")
        queue.enqueue(conn, "train", job_id)


def stop(conn, job_id: int) -> str:
    """A queued job stops at once; a running one at its next check, between attempts."""
    with conn.transaction():
        job = conn.execute("SELECT status FROM jobs WHERE id = %s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["status"] not in ("queued", "running"):
            raise SystemExit(f"job {job_id} is not queued or running")
        cancelled = conn.execute(
            "UPDATE tasks SET status = 'cancelled' WHERE job_id = %s AND status = 'queued' RETURNING id", (job_id,)
        ).fetchone()
        status = "stopped" if cancelled else "stopping"
        conn.execute(
            "UPDATE jobs SET status = %s, cancel_requested = %s WHERE id = %s", (status, not cancelled, job_id)
        )
    return status


def resume(conn, job_id: int) -> str:
    """Carry a stopped or failed job on from where its thread is: plan again if it never got a plan."""
    with conn.transaction():
        job = conn.execute("SELECT status, plan, files_expired FROM jobs WHERE id = %s FOR UPDATE", (job_id,)).fetchone()
        if job is None or job["status"] not in ("stopped", "failed"):
            raise SystemExit(f"job {job_id} is not stopped or failed")
        if job["files_expired"]:
            raise SystemExit(f"job {job_id}'s files have expired")
        kind = "plan" if job["plan"] is None else "train"
        status = "planning" if kind == "plan" else "queued"
        conn.execute(
            "UPDATE jobs SET status = %s, cancel_requested = false, error = NULL, finished_at = NULL, "
            "files_expire_at = NULL WHERE id = %s",
            (status, job_id),
        )
        queue.enqueue(conn, kind, job_id)
    return status


# ---- the commands

def _enqueue(conn, args) -> None:
    from try_out_runner import RUNS

    spec = next((run for run in RUNS if run["slug"] == args.slug), None)
    if spec is None:
        raise SystemExit(f"unknown dataset {args.slug!r}; one of: {', '.join(run['slug'] for run in RUNS)}")
    job_id = create_job(conn, workspace(conn, args.workspace), spec["topic"], spec["data_path"], spec["schema"])
    print(f"job {job_id}: planning {args.slug}")


def _add_host(conn, args) -> None:
    from code_gen_eval.runner import RUNNER_IMAGE
    from code_gen_eval.sandbox_client import SandboxClient, SandboxError, SandboxUnavailable

    token = os.environ.get(args.token_env) if args.token_env else getpass.getpass("sandbox host token: ")
    if not token:
        raise SystemExit(f"{args.token_env} is empty")
    try:
        info = SandboxClient(args.url, token).info(retry=False)   # proves the link and the token before storing them
    except (SandboxError, SandboxUnavailable) as error:
        raise SystemExit(f"not added: {error}")
    runtime_hash = info.get("images", {}).get(RUNNER_IMAGE, {}).get("runtime_hash")
    workspace_id = workspace(conn, args.workspace) if args.workspace else None
    host_id = add_host(conn, args.url, token, crypto.secret_key(), workspace_id, args.max_running_jobs, runtime_hash)
    owner = f"workspace {args.workspace}" if args.workspace else "the platform"
    print(f"host {host_id}: {args.url} for {owner}, token ...{crypto.last4(token)}, runtime {runtime_hash}")


def _status(conn, args) -> None:
    if args.job is None:
        for job in conn.execute(
            "SELECT j.id, w.slug, j.status, j.name, j.created_at FROM jobs j JOIN workspaces w ON w.id = j.workspace_id "
            "ORDER BY j.id DESC LIMIT 20"
        ):
            print(f"{job['id']:>5}  {job['slug']:<12} {job['status']:<10} {job['created_at']:%Y-%m-%d %H:%M}  {job['name']}")
        return

    job = conn.execute("SELECT * FROM jobs WHERE id = %s", (args.job,)).fetchone()
    if job is None:
        raise SystemExit(f"no job {args.job}")
    print(f"job {job['id']}: {job['status']}  host {job['sandbox_host_id']}  run_dir {job['run_dir']}")
    if job["error"]:
        print(f"error: {job['error']}")
    if job["status"] == "review" and job["plan"]:
        config, split = job["plan"]["config"], job["plan"]["split_plan"]
        print(f"plan: models {', '.join(config['models'])}; metrics {', '.join(config['eval_matrics'])}; "
              f"split {split['method']} test {split['test_size']}, {split['cv_strategy']} x{split['cv_folds']}")
    for model in conn.execute("SELECT * FROM job_models WHERE job_id = %s ORDER BY model", (args.job,)):
        print(f"  {model['model']:<24} {model['status']:<10} gen {model['generation']}  best {model['best_cv']}")
    usage = conn.execute(
        "SELECT meter, unit, sum(quantity) AS total FROM usage_records WHERE job_id = %s GROUP BY meter, unit "
        "ORDER BY meter",
        (args.job,),
    ).fetchall()
    if usage:
        print("usage: " + ", ".join(f"{row['meter']} {row['total']:.4g} {row['unit']}" for row in usage))
    events = conn.execute(
        "SELECT at, kind, model, data FROM job_events WHERE job_id = %s ORDER BY id DESC LIMIT 10", (args.job,)
    ).fetchall()
    for event in reversed(events):
        print(f"  {event['at']:%H:%M:%S} {event['kind']:<18} {event['model'] or '':<20} {json.dumps(event['data'])[:90]}")


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m worker.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add-host", help="register a sandbox server")
    add.add_argument("url")
    add.add_argument("--workspace", help="the only workspace whose jobs it runs (default: the platform's own)")
    add.add_argument("--token-env", help="read the token from this environment variable instead of asking")
    add.add_argument("--max-running-jobs", type=int, default=1)

    set_host = commands.add_parser("set-host", help="change how many jobs a sandbox host trains at once")
    set_host.add_argument("host", type=int)
    set_host.add_argument("--max-running-jobs", type=int, required=True)

    enqueue = commands.add_parser("enqueue", help="plan a try-out dataset")
    enqueue.add_argument("slug")
    enqueue.add_argument("--workspace", default=DEFAULT_WORKSPACE)

    approve_cmd = commands.add_parser("approve", help="approve a plan, with optional edits, and queue training")
    approve_cmd.add_argument("job", type=int)
    approve_cmd.add_argument("--edit", type=_json_arg, default={},
                             help='{"split_plan": {...}, "config": {...}}, or a file holding it')

    for name, text in (("stop", "stop a queued or running job"), ("resume", "carry a stopped or failed job on")):
        command = commands.add_parser(name, help=text)
        command.add_argument("job", type=int)

    status = commands.add_parser("status", help="list jobs, or show one")
    status.add_argument("job", type=int, nargs="?")

    for name, text in (("allow", "let these emails sign up"), ("disallow", "take these emails off the allowlist")):
        command = commands.add_parser(name, help=text)
        command.add_argument("emails", nargs="+", metavar="EMAIL")

    args = parser.parse_args(argv)
    pool = db.pool(size=2)
    try:
        with pool.connection() as conn:
            db.migrate(conn)   # the checkpoint tables are the worker's to create
            if args.command == "add-host":
                _add_host(conn, args)
            elif args.command == "set-host":
                row = conn.execute("UPDATE sandbox_hosts SET max_running_jobs = %s WHERE id = %s RETURNING url",
                                   (args.max_running_jobs, args.host)).fetchone()
                if row is None:
                    raise SystemExit(f"no host {args.host}")
                print(f"host {args.host} ({row['url']}): up to {args.max_running_jobs} jobs at once")
            elif args.command == "enqueue":
                _enqueue(conn, args)
            elif args.command == "approve":
                approve(conn, args.job, args.edit)
                print(f"job {args.job}: queued")
            elif args.command == "stop":
                print(f"job {args.job}: {stop(conn, args.job)}")
            elif args.command == "resume":
                print(f"job {args.job}: {resume(conn, args.job)}")
            elif args.command in ("allow", "disallow"):
                (allow if args.command == "allow" else disallow)(conn, args.emails)
                emails = [row["email"] for row in conn.execute("SELECT email FROM signup_allowlist ORDER BY email")]
                print(f"allowlist ({len(emails)}): {', '.join(emails) or 'empty'}")
            else:
                _status(conn, args)
    finally:
        pool.close()


if __name__ == "__main__":
    main(sys.argv[1:])
