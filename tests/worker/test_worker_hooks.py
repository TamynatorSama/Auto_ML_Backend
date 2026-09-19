import pytest

from worker import hooks as worker_hooks
from worker import jobs
from worker.hooks import WorkerHooks

USAGE = {"wall_seconds": 11.2, "cpu_seconds": 12.5, "peak_memory_mb": 193.0, "reserved_gb_seconds": 45.9,
         "reserved_cpu_seconds": 22.4, "measured_with": "cgroup-v2"}


@pytest.fixture
def worker(pool, key):
    return WorkerHooks(pool, key)


def _one_attempt(worker, job_id, attempt):
    worker.emit(job_id, "sandbox_usage", model="ridge", attempt=str(attempt), step="loop", usage=USAGE)
    worker.emit(job_id, "attempt_finished", model="ridge", attempt=attempt, generation=1, attempt_kind="generate",
                status="ok", changes="", cv_scores={"rmse": 1.5}, wall_seconds=11.2, usage=USAGE)


def test_events_land_in_their_tables(pool, worker, new_job):
    job_id = new_job()
    worker.emit(job_id, "model_waiting", model="ridge")
    worker.emit(job_id, "model_started", model="ridge")
    worker.emit(job_id, "generation_started", model="ridge", generation=2, attempt=3)
    _one_attempt(worker, job_id, 1)
    worker.emit(job_id, "model_finished", model="ridge", status="ok", best_attempt=1, best_cv_scores={"rmse": 1.5},
                test_scores={"rmse": float("nan")}, eligibility="eligible", error="")

    with pool.connection() as conn:
        kinds = [row["kind"] for row in conn.execute("SELECT kind FROM job_events WHERE job_id = %s ORDER BY id", (job_id,))]
        model = conn.execute("SELECT * FROM job_models WHERE job_id = %s", (job_id,)).fetchone()
        attempt = conn.execute("SELECT * FROM job_attempts WHERE job_id = %s", (job_id,)).fetchone()
        meters = {row["meter"]: row for row in conn.execute("SELECT * FROM usage_records WHERE job_id = %s", (job_id,))}

    assert kinds == ["model_waiting", "model_started", "generation_started", "sandbox_usage", "attempt_finished",
                     "model_finished"]
    assert (model["status"], model["generation"], model["best_attempt"]) == ("ok", 2, 1)
    assert model["test_scores"] == {"rmse": None}   # jsonb has no NaN
    assert (attempt["attempt"], attempt["status"], attempt["cv_scores"]) == (1, "ok", {"rmse": 1.5})
    assert set(meters) == set(worker_hooks.SANDBOX_METERS)
    assert (meters["cpu_seconds"]["model"], meters["cpu_seconds"]["attempt"]) == ("ridge", 1)
    assert meters["cpu_seconds"]["sandbox_host_id"] is not None


def test_usage_totals_are_the_sum_of_the_ledger_and_of_the_attempts(pool, worker, new_job):
    job_id = new_job()
    for attempt in (1, 2, 3):
        _one_attempt(worker, job_id, attempt)
    worker.record_usage(job_id, "judge", "gemini-2.5-flash", 1000, 200)
    worker.record_usage(job_id, "fixer", "gemini-2.5-flash", 500, 100)

    with pool.connection() as conn:
        jobs.total_usage(conn, job_id)
        totals = conn.execute("SELECT usage_totals FROM jobs WHERE id = %s", (job_id,)).fetchone()["usage_totals"]
        attempts_cpu = conn.execute(
            "SELECT sum((usage->>'cpu_seconds')::float) AS cpu FROM job_attempts WHERE job_id = %s", (job_id,)
        ).fetchone()["cpu"]
        roles = {row["role"] for row in conn.execute("SELECT role FROM usage_records WHERE source = 'llm'")}

    assert totals["cpu_seconds"] == pytest.approx(3 * 12.5) == pytest.approx(attempts_cpu)
    assert totals["reserved_gb_seconds"] == pytest.approx(3 * 45.9)
    assert (totals["input_tokens"], totals["output_tokens"]) == (1500, 300)
    assert totals["cost_usd"] > 0
    assert roles == {"judge", "fixer"}


def test_a_stop_requested_in_the_database_stops_the_run(pool, worker, new_job, monkeypatch):
    monkeypatch.setattr(worker_hooks, "STOP_CHECK_SECONDS", 0)
    job_id = new_job()
    assert worker.should_stop(job_id) is None
    with pool.connection() as conn:
        conn.execute("UPDATE jobs SET cancel_requested = true WHERE id = %s", (job_id,))
    assert worker.should_stop(job_id) == "stop requested"


def test_each_job_gets_a_client_for_its_own_host(pool, worker, new_job, key):
    from worker import cli

    platform_job = new_job()
    with pool.connection() as conn:
        tester = cli.workspace(conn, "tester")
        cli.add_host(conn, "http://tester-sandbox.test", "tester-token-9876", key, tester)
    tester_job = new_job("tester")

    platform, own = worker.sandbox_for(platform_job), worker.sandbox_for(tester_job)
    assert (platform.url, platform.http.headers["Authorization"]) == ("http://sandbox.test", "Bearer a-token-1234")
    assert (own.url, own.http.headers["Authorization"]) == ("http://tester-sandbox.test", "Bearer tester-token-9876")
    assert worker.sandbox_for(tester_job) is own

    own.on_wait("host unreachable", 5)
    worker.emit(tester_job, "sandbox_usage", model="ridge", attempt="1", step="loop", usage=USAGE)
    with pool.connection() as conn:
        waiting = conn.execute("SELECT data FROM job_events WHERE kind = 'sandbox_waiting'").fetchone()["data"]
        hosts = {row["sandbox_host_id"] for row in conn.execute(
            "SELECT sandbox_host_id FROM usage_records WHERE job_id = %s", (tester_job,))}
        tester_host = conn.execute("SELECT sandbox_host_id FROM jobs WHERE id = %s", (tester_job,)).fetchone()
    assert waiting == {"problem": "host unreachable", "retry_in": 5}
    assert hosts == {tester_host["sandbox_host_id"]}

    worker.forget(tester_job)
    assert worker.sandbox_for(tester_job) is not own


def test_an_unreachable_database_stops_the_model_instead_of_failing_it(worker, new_job, monkeypatch):
    import psycopg
    from utils.reusable.hooks import RunStopped

    job_id = new_job()
    worker._job(job_id)   # cached before the database goes away
    monkeypatch.setattr(worker_hooks, "STOP_CHECK_SECONDS", 0)

    def down(*args, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(worker.pool, "connection", down)
    with pytest.raises(RunStopped, match="database unavailable"):
        worker.emit(job_id, "model_started", model="ridge")
    assert worker.should_stop(job_id) == "database unavailable"


def test_jobs_on_one_host_share_one_pool(pool, worker, new_job, key, monkeypatch):
    from types import SimpleNamespace
    from worker import cli

    hosts = []
    monkeypatch.setattr(worker, "sandbox_for", lambda run_id: (
        hosts.append(run_id), SimpleNamespace(capacity=lambda: {"memory_budget_mb": 8192, "cpus": 8}))[1])
    first, second = new_job(), new_job("other")
    with pool.connection() as conn:
        own = cli.add_host(conn, "http://own-sandbox.test", "own-token-1111", key, cli.workspace(conn, "own"))
    third = new_job("own")

    with worker.sandbox_slot(first, 2048, 2):
        with worker.sandbox_slot(second, 2048, 2):
            shared = worker._pools[worker._job(first)["sandbox_host_id"]]
            assert shared.used_memory == 4096 and dict(shared.running) == {first: 1, second: 1}
            assert shared.cpus == 6   # 2 of the host's 8 stay free
        with worker.sandbox_slot(third, 2048, 2):
            assert worker._pools[own].running == {third: 1} and shared.running == {first: 1}
    assert len(worker._pools) == 2
