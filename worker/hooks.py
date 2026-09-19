"""
hooks.py
--------
The pipeline's hooks for jobs run by the worker (docs/PHASE2.md §6). The run id
is the job id.

    emit         every event into job_events, and the derived tables kept current:
                 attempt_finished -> job_attempts, model_* -> job_models,
                 sandbox_usage and llm_usage -> usage_records (the ledger, D10)
    should_stop  the defaults (budget, deadline), plus jobs.cancel_requested
    sandbox_for  a client for the job's own sandbox host (D11), cached per job
    sandbox_slot a place in the fair pool of the job's host, shared with every other
                 job training there (docs/PHASE2B.md)
    llm_for      unchanged: Gemini from GOOGLE_API_KEY until Phase 5
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import psycopg
from psycopg_pool import ConnectionPool

from code_gen_eval.sandbox_client import SandboxClient
from db import jsonb
from utils.reusable.hooks import Hooks, RunStopped
from worker import crypto
from worker.pool import HostPool, usable_cpus

STOP_CHECK_SECONDS = 5
# meter -> unit
SANDBOX_METERS = {"cpu_seconds": "s", "reserved_gb_seconds": "GB-s", "reserved_cpu_seconds": "CPU-s", "wall_seconds": "s"}
LLM_METERS = {"input_tokens": "tokens", "output_tokens": "tokens", "cost_usd": "USD"}
MODEL_STATUS = {"model_waiting": "waiting", "model_started": "running"}


def _int(value) -> Optional[int]:
    return int(value) if str(value).isdigit() else None


class WorkerHooks(Hooks):
    def __init__(self, pool: ConnectionPool, secret_key: str):
        super().__init__()
        self.pool, self.secret_key = pool, secret_key
        self._cache_lock = threading.Lock()
        self._jobs: dict = {}       # job id -> workspace_id, sandbox_host_id
        self._clients: dict = {}    # job id -> SandboxClient
        self._stop_checked: dict = {}   # job id -> (monotonic time, cancel_requested)
        self._pools: dict = {}      # host id -> HostPool, for the worker's lifetime

    def _job(self, job_id) -> dict:
        with self._cache_lock:
            if job_id not in self._jobs:
                with self.pool.connection() as conn:
                    self._jobs[job_id] = conn.execute(
                        "SELECT workspace_id, sandbox_host_id FROM jobs WHERE id = %s", (job_id,)
                    ).fetchone()
            return self._jobs[job_id]

    def forget(self, job_id) -> None:
        """Drop what was cached for a job once its task ends."""
        with self._cache_lock:
            self._jobs.pop(job_id, None)
            self._stop_checked.pop(job_id, None)
            client = self._clients.pop(job_id, None)
        if client is not None:
            client.http.close()

    # ---- events

    def emit(self, run_id, kind: str, **data) -> None:
        try:
            self._record(run_id, kind, data)
        except psycopg.OperationalError as error:
            # unreachable (the tunnel is down): stop the model, resumably, rather than fail it
            raise RunStopped(f"database unavailable: {error}") from error

    def _record(self, run_id, kind: str, data: dict) -> None:
        job = self._job(run_id)
        with self.pool.connection() as conn, conn.transaction():
            model = None if kind == "llm_usage" else data.get("model")
            conn.execute(
                "INSERT INTO job_events (job_id, kind, model, data) VALUES (%s, %s, %s, %s)",
                (run_id, kind, model, jsonb(data)),
            )
            if kind == "attempt_finished":
                self._attempt(conn, run_id, data)
            elif kind in MODEL_STATUS or kind in ("model_resumed", "model_finished"):
                self._model(conn, run_id, MODEL_STATUS.get(kind) or data["status"], data)
            elif kind == "generation_started":
                conn.execute(
                    "UPDATE job_models SET generation = %s WHERE job_id = %s AND model = %s",
                    (data["generation"], run_id, data["model"]),
                )
            elif kind == "sandbox_usage":
                usage = data.get("usage") or {}
                self._ledger(conn, run_id, job, "sandbox", SANDBOX_METERS, usage,
                             model=data.get("model"), attempt=_int(data.get("attempt")))
            elif kind == "llm_usage":
                self._ledger(conn, run_id, job, "llm", LLM_METERS, data, model=data.get("model"), role=data.get("role"))

    def _attempt(self, conn, job_id, data: dict) -> None:
        conn.execute(
            """
            INSERT INTO job_attempts (job_id, model, attempt, generation, kind, status, cv_scores,
                                      wall_seconds, changes, usage)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (job_id, data["model"], data["attempt"], data.get("generation"), data.get("attempt_kind"),
             data["status"], jsonb(data.get("cv_scores") or {}), data.get("wall_seconds"),
             data.get("changes"), jsonb(data.get("usage") or {})),
        )

    def _model(self, conn, job_id, status: str, data: dict) -> None:
        conn.execute(
            """
            INSERT INTO job_models (job_id, model, status, best_attempt, best_cv, test_scores)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id, model) DO UPDATE SET
                status = EXCLUDED.status,
                best_attempt = coalesce(EXCLUDED.best_attempt, job_models.best_attempt),
                best_cv = coalesce(EXCLUDED.best_cv, job_models.best_cv),
                test_scores = coalesce(EXCLUDED.test_scores, job_models.test_scores)
            """,
            (job_id, data["model"], status, data.get("best_attempt"),
             jsonb(data["best_cv_scores"]) if "best_cv_scores" in data else None,
             jsonb(data["test_scores"]) if "test_scores" in data else None),
        )

    def _ledger(self, conn, job_id, job: dict, source: str, meters: dict, values: dict,
                model=None, attempt=None, role=None) -> None:
        rows = [
            (job["workspace_id"], job_id, job["sandbox_host_id"], source, meter, float(values[meter]), unit,
             model, attempt, role)
            for meter, unit in meters.items() if values.get(meter) is not None
        ]
        if rows:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO usage_records (workspace_id, job_id, sandbox_host_id, source, meter, quantity,
                                               unit, model, attempt, role)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )

    # ---- stop

    def should_stop(self, run_id) -> Optional[str]:
        reason = super().should_stop(run_id)
        if reason:
            return reason
        checked_at, requested = self._stop_checked.get(run_id, (float("-inf"), False))
        if time.monotonic() - checked_at >= STOP_CHECK_SECONDS:
            try:
                with self.pool.connection() as conn:
                    row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = %s", (run_id,)).fetchone()
            except psycopg.OperationalError:
                return "database unavailable"
            requested = bool(row and row["cancel_requested"])
            self._stop_checked[run_id] = (time.monotonic(), requested)
        return "stop requested" if requested else None

    # ---- the job's sandbox host

    def sandbox_for(self, run_id):
        with self._cache_lock:
            if run_id not in self._clients:
                with self.pool.connection() as conn:
                    host = conn.execute(
                        "SELECT h.url, h.token_ciphertext FROM jobs j "
                        "JOIN sandbox_hosts h ON h.id = j.sandbox_host_id WHERE j.id = %s",
                        (run_id,),
                    ).fetchone()
                token = crypto.decrypt(self.secret_key, host["token_ciphertext"])
                self._clients[run_id] = SandboxClient(
                    host["url"], token,
                    on_wait=lambda problem, delay: self.emit(run_id, "sandbox_waiting", problem=problem, retry_in=delay),
                )
            return self._clients[run_id]

    def sandbox_slot(self, run_id, memory_mb: float, cpus: float):
        host_id = self._job(run_id)["sandbox_host_id"]
        if host_id not in self._pools:
            capacity = self.sandbox_for(run_id).capacity()   # outside the lock: sandbox_for takes it
            with self._cache_lock:
                self._pools.setdefault(host_id, HostPool(capacity["memory_budget_mb"], usable_cpus(capacity["cpus"])))
        return self._pools[host_id].slot(run_id, memory_mb, cpus, lambda: self.should_stop(run_id))
