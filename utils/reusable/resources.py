"""
resources.py
------------
How many models can train at once, with how many threads each, on this machine.

A fixed setting cannot be right for every dataset. Two workers with two threads
each were fine on 14,000 rows and ran a 16 GB machine out of memory on 306,000,
and the crash took every model with it. The plan is made from the machine as it
is when the run starts (free memory, cores) and the size of the training data,
and it is recorded in the run so the choice can be read afterwards.

    plan_resources(train_path, n_models) -> ResourcePlan
    wait_for_memory(needed_mb)           -> seconds waited
    memory_limit(worker_mb, concurrency) -> MB one evaluator may use before it is stopped
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
PARALLEL_WORKER_MB = 170      # a joblib worker process: its own interpreter with pandas and scikit-learn
MEMORY_WAIT_SECONDS = 600
MEMORY_POLL_SECONDS = 2


@dataclass
class ResourcePlan:
    max_concurrency: int
    n_jobs: int
    worker_memory_mb: float
    available_memory_mb: float
    cpus: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _usable_mb() -> float:
    available = psutil.virtual_memory().available / 1e6
    return max(0.0, available - SYSTEM_RESERVE_MB) * MEMORY_HEADROOM


def plan_resources(train_path: str, n_models: int) -> ResourcePlan:
    cpus = os.cpu_count() or 2
    available = psutil.virtual_memory().available / 1e6
    frame = Path(train_path).stat().st_size / 1e6 * DATA_EXPANSION
    per_worker = WORKER_OVERHEAD_MB + frame * WORKING_COPIES

    usable = _usable_mb()
    by_memory = int(usable // per_worker)
    by_cores = max(1, cpus // 2)
    concurrency = max(1, min(max(1, n_models), by_cores, by_memory))

    # n_jobs is often a pool of processes (a ColumnTransformer's encoders, a
    # grid search), each loading its own libraries: parallelism is paid for in
    # memory, so it is granted only where the memory is there to pay for it
    share = usable / concurrency
    by_memory_jobs = 1 + int(max(0.0, share - per_worker) // PARALLEL_WORKER_MB)
    n_jobs = max(1, min(MAX_THREADS, cpus // concurrency, by_memory_jobs))
    per_worker += (n_jobs - 1) * PARALLEL_WORKER_MB

    limit = "memory" if by_memory <= min(by_cores, n_models) else ("cores" if by_cores < n_models else "models")
    jobs_limit = "memory" if by_memory_jobs < min(MAX_THREADS, cpus // concurrency) else "cores"
    reason = (
        f"{available:.0f} MB free and about {per_worker:.0f} MB per worker for {frame:.0f} MB of data; "
        f"{cpus} cores; {concurrency} worker(s), limited by {limit}, with n_jobs {n_jobs}, limited by {jobs_limit}"
    )
    return ResourcePlan(concurrency, n_jobs, round(per_worker, 1), round(available, 1), cpus, reason)


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
