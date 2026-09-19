import threading

from worker import cli, db, queue


def test_migrations_apply_once(pool):
    with pool.connection() as conn:
        assert db.migrate(conn) == []   # the fixture already applied them
        names = [row["name"] for row in conn.execute("SELECT name FROM schema_migrations")]
    assert names == ["001_initial.sql"]


def test_tasks_are_claimed_oldest_first(pool, new_job):
    first, second = new_job(), new_job()
    with pool.connection() as conn:
        claimed = [queue.claim(conn)["job_id"], queue.claim(conn)["job_id"]]
        assert queue.claim(conn) is None
    assert claimed == [first, second]


def _train_queued(pool, job_id):
    with pool.connection() as conn:
        conn.execute("UPDATE tasks SET status = 'done' WHERE job_id = %s", (job_id,))
        conn.execute("UPDATE jobs SET status = 'review' WHERE id = %s", (job_id,))
        cli.approve(conn, job_id)


def test_a_host_runs_one_training_at_a_time_but_planning_goes_ahead(pool, new_job):
    first, second = new_job(), new_job()
    _train_queued(pool, first)
    _train_queued(pool, second)
    third = new_job()   # its plan task is queued after both train tasks

    with pool.connection() as conn:
        running = queue.claim(conn)
        assert (running["kind"], running["job_id"]) == ("train", first)
        planning = queue.claim(conn)
        assert (planning["kind"], planning["job_id"]) == ("plan", third)   # second waits its turn
        assert queue.claim(conn) is None

        queue.finish(conn, running["id"])
        assert queue.claim(conn)["job_id"] == second


def test_two_claims_at_once_cannot_both_see_room(pool, new_job):
    jobs = [new_job(), new_job()]
    for job_id in jobs:
        _train_queued(pool, job_id)

    barrier, claimed = threading.Barrier(2), []

    def claim():
        with pool.connection() as conn:
            barrier.wait()
            claimed.append(queue.claim(conn))

    threads = [threading.Thread(target=claim) for _ in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(task is not None for task in claimed) == 1


def _go_stale(pool, task_id):
    with pool.connection() as conn:
        conn.execute("UPDATE tasks SET heartbeat_at = now() - interval '2 minutes' WHERE id = %s", (task_id,))


def test_a_stale_task_is_requeued_and_fails_its_job_after_three_tries(pool, new_job):
    job_id = new_job()
    with pool.connection() as conn:
        for attempt in range(1, queue.MAX_TRIES + 1):
            task = queue.claim(conn)
            assert task["tries"] == attempt
            queue.heartbeat(conn, task["id"])
            assert queue.requeue_stale(conn) == []   # a fresh heartbeat is left alone
            _go_stale(pool, task["id"])
            expected = "failed" if attempt == queue.MAX_TRIES else "queued"
            assert queue.requeue_stale(conn) == [(task["id"], expected)]

        job = conn.execute("SELECT status, error, files_expire_at FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert job["status"] == "failed" and "3 times" in job["error"] and job["files_expire_at"] is not None


def test_stop_cancels_a_queued_job_and_flags_a_running_one(pool, new_job):
    queued, running = new_job(), new_job()
    for job_id in (queued, running):
        _train_queued(pool, job_id)
    with pool.connection() as conn:
        conn.execute("UPDATE tasks SET status = 'running' WHERE job_id = %s AND kind = 'train'", (running,))
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = %s", (running,))
        assert cli.stop(conn, queued) == "stopped"
        assert cli.stop(conn, running) == "stopping"
        jobs = {row["id"]: (row["status"], row["cancel_requested"]) for row in conn.execute("SELECT * FROM jobs")}
        tasks = {row["job_id"]: row["status"] for row in conn.execute("SELECT job_id, status FROM tasks WHERE kind = 'train'")}
    assert jobs[queued] == ("stopped", False) and tasks[queued] == "cancelled"
    assert jobs[running] == ("stopping", True) and tasks[running] == "running"


def test_several_jobs_train_at_once_but_one_per_workspace(pool, new_job):
    jobs = {name: new_job(name) for name in ("ann", "bob", "cy")}
    ann_second = new_job("ann")
    for job_id in [*jobs.values(), ann_second]:
        _train_queued(pool, job_id)
    with pool.connection() as conn:
        conn.execute("UPDATE sandbox_hosts SET max_running_jobs = 3")
        claimed = [queue.claim(conn) for _ in range(4)]

    assert [task["job_id"] for task in claimed[:3]] == list(jobs.values())
    assert claimed[3] is None   # ann's second job waits for her first, though the host has no room anyway
    with pool.connection() as conn:
        conn.execute("UPDATE sandbox_hosts SET max_running_jobs = 4")
        assert queue.claim(conn) is None   # room on the host now, but ann already has one training
        queue.finish(conn, claimed[0]["id"])
        assert queue.claim(conn)["job_id"] == ann_second


def test_a_worker_can_take_only_training(pool, new_job):
    planning, training = new_job(), new_job("other")
    _train_queued(pool, training)
    with pool.connection() as conn:
        assert queue.claim(conn, ["train"])["job_id"] == training
        assert queue.claim(conn, ["train"]) is None
        assert queue.claim(conn)["job_id"] == planning
