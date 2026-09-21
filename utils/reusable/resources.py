"""
resources.py
------------
How many models can train at once, with how many threads each, on this machine.

A fixed setting cannot be right for every dataset. Two workers with two threads
each were fine on 14,000 rows and ran a 16 GB machine out of memory on 306,000,
and the crash took every model with it. The plan is made from the machine as it
is when the run starts (free memory, cores) and the size of the training data,
and it is recorded in the run so the choice can be read afterwards.

    plan_resources(train_path, n_models, client=None) -> ResourcePlan
    usable_cpus(cpus)                    -> what sandboxes may reserve on a host
    wait_for_memory(needed_mb)           -> seconds waited
    memory_limit(worker_mb, concurrency) -> MB one evaluator may use before it is stopped

A sandbox is given a ceiling and charged a reservation, and they are not the
same number. The ceiling is the cgroup limit: exceed it and the kernel kills the
process, so it is generous. The reservation is what the worker's pool counts
against the host while the sandbox runs, so it is what the estimate says the
attempt will actually need. Charging the ceiling let one sandbox reserve the
host's whole memory budget and half its cores while using an eighth of the
memory and two thirds of the cores.

Cores are the looser of the two. Memory over its limit is a kill, so the ceiling
has to be pessimistic; CPU over its share is only slowness, so the host runs
sandboxes on a weight rather than a private slice and an idle neighbour's cores
go to whoever can use them. So an attempt reserves the one thread it is certain
to keep busy (CPU_RESERVE) and may burst to the host's per-sandbox cap. n_jobs
stays tied to the share it can count on when every sandbox is busy: the ceiling
is room to burst, not a licence to spawn threads for cores nobody has.

With a sandbox client the plan is made from the sandbox host's memory budget
instead of this machine, and worker_memory_mb is the memory each sandbox gets.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

WORKER_OVERHEAD_MB = 350      # an interpreter with pandas, scikit-learn and a booster loaded
DATA_EXPANSION = 2.0          # a CSV on disk to a DataFrame in memory
WORKING_COPIES = 4            # a fold's copies, encodings and the model's own working memory
MEMORY_HEADROOM = 0.8         # leave part of what is free to the rest of the machine
SYSTEM_RESERVE_MB = 1_000     # and never plan on the last gigabyte
MAX_THREADS = 4
CPU_RESERVE = 1.0             # the one thread of an attempt that is always runnable
PARALLEL_WORKER_MB = 170      # a joblib worker process: its own interpreter with pandas and scikit-learn
# the reservation is the estimate with room to be wrong: measured over 26 sandboxes,
# the estimate landed near the average peak and 1.5x covered the worst of them
RESERVE_MARGIN = 1.5
MEMORY_WAIT_SECONDS = 600
MEMORY_POLL_SECONDS = 2


@dataclass
class ResourcePlan:
    max_concurrency: int
    n_jobs: int
    worker_memory_mb: float          # the ceiling: what the sandbox is killed above
    available_memory_mb: float
    cpus: int
    reason: str
    reserve_memory_mb: float = 0.0   # what the pool counts while it runs
    reserve_cpus: float = CPU_RESERVE

    def to_dict(self) -> dict:
        return asdict(self)


def usable_cpus(cpus: float) -> float:
    """What sandboxes may reserve on a host with this many CPUs.

    A quarter (at least one) stays free for gVisor, the sandbox server and the rest of
    the machine: with all 8 of 8 reserved, attempts ran about twice as slow and one
    timed out. The planner divides by this and the pool admits against it, so the
    concurrency a plan asks for is one the pool can actually grant.
    """
    return max(1.0, cpus - max(1.0, cpus // 4))


def _usable_mb() -> float:
    available = psutil.virtual_memory().available / 1e6
    return max(0.0, available - SYSTEM_RESERVE_MB) * MEMORY_HEADROOM


def plan_resources(train_path: str, n_models: int, client=None) -> ResourcePlan:
    frame = Path(train_path).stat().st_size / 1e6 * DATA_EXPANSION
    per_worker = WORKER_OVERHEAD_MB + frame * WORKING_COPIES
    max_threads = MAX_THREADS

    if client is not None:
        # the host's whole budget, not what is free right now: jobs share the host, and
        # the worker queues their sandboxes fairly, so a job planned while others run
        # gets the same sandboxes as one planned alone
        capacity, limits = client.capacity(), client.info()["limits"]
        # what the pool will admit, not what the host has: planning for all of them
        # asked for a concurrency the pool could never grant, so nothing ran in parallel
        cpus = int(usable_cpus(capacity["cpus"]))
        available = usable = float(capacity["memory_budget_mb"])
        max_threads = min(MAX_THREADS, int(limits["max_cpus"]))
        where = "in the sandbox budget"
    else:
        cpus = os.cpu_count() or 2
        available = psutil.virtual_memory().available / 1e6
        usable = _usable_mb()
        where = "free"

    by_memory = int(usable // per_worker)
    # one runnable core per attempt, not a private pair: the sandbox's cores are a
    # weight now, so an attempt that wants more takes it from whoever is idle
    by_cores = max(1, int(cpus))
    concurrency = max(1, min(max(1, n_models), by_cores, by_memory))

    # n_jobs is often a pool of processes (a ColumnTransformer's encoders, a
    # grid search), each loading its own libraries: parallelism is paid for in
    # memory, so it is granted only where the memory is there to pay for it
    share = usable / concurrency
    by_memory_jobs = 1 + int(max(0.0, share - per_worker) // PARALLEL_WORKER_MB)
    n_jobs = max(1, min(max_threads, cpus // concurrency, by_memory_jobs))
    per_worker += (n_jobs - 1) * PARALLEL_WORKER_MB

    reserve = min(usable, per_worker * RESERVE_MARGIN)
    if client is not None:
        # the ceiling is its share of the budget, as memory_limit gives an evaluator here:
        # generous, because the kernel kills whatever passes it. The reservation above
        # stays at the estimate, so a generous ceiling costs no concurrency.
        per_worker = min(float(limits["max_sandbox_memory_mb"]), max(per_worker, share))
        reserve = min(reserve, per_worker)

    limit = "memory" if by_memory <= min(by_cores, n_models) else ("cores" if by_cores < n_models else "models")
    jobs_limit = "memory" if by_memory_jobs < min(max_threads, cpus // concurrency) else "cores"
    reason = (
        f"{available:.0f} MB {where} and about {per_worker:.0f} MB per worker for {frame:.0f} MB of data; "
        f"{cpus} cores; {concurrency} worker(s), limited by {limit}, with n_jobs {n_jobs}, limited by {jobs_limit}; "
        f"each reserves {reserve:.0f} MB and {CPU_RESERVE:g} core of the host while it runs"
    )
    return ResourcePlan(concurrency, n_jobs, round(per_worker, 1), round(available, 1), cpus, reason,
                        round(reserve, 1), CPU_RESERVE)


def wait_for_memory(needed_mb: float) -> float:
    """Hold off starting a process until the machine has room for it.

    A plan made at the start cannot see memory other applications take later.
    Waiting is better than starting a fit that will be killed; after a while the
    process starts anyway, because an estimate should not stall a run forever.
    """
    if needed_mb <= 0:
        return 0.0
    started = time.monotonic()
    while psutil.virtual_memory().available / 1e6 < needed_mb:
        if time.monotonic() - started > MEMORY_WAIT_SECONDS:
            break
        time.sleep(MEMORY_POLL_SECONDS)
    return time.monotonic() - started


def memory_limit(worker_mb: float, concurrency: int) -> float:
    """How much one evaluator may use before it is stopped.

    What a model needs depends on the model, which no plan made from the data
    can see: fully grown trees on 306,000 rows took gigabytes where a booster
    took 400 MB. So the plan sets how many run at once, and this sets a ceiling
    each one is held to, from what is free when it starts, shared between the
    evaluators that may be running. Never below the plan's own estimate.
    """
    return max(worker_mb, _usable_mb() / max(1, concurrency))
