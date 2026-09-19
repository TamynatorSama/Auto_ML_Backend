"""
pool.py
-------
One sandbox host's memory and CPUs, shared fairly between the jobs training on
it (docs/PHASE2B.md §3). A sandbox run asks for its memory and CPUs and waits in
line; when places free up, the run whose job has the fewest sandboxes running
goes first, and between runs of one job, the one that asked first. The first in
line is never skipped for a smaller run behind it, so a large run can't starve.

The host's own 429 stays as the safety net for anything outside this process.
"""

from __future__ import annotations

import itertools
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Callable, Optional

from utils.reusable.hooks import RunStopped

WAIT_SECONDS = 1   # how often a run in line checks for a stop


def usable_cpus(cpus: float) -> float:
    """What sandboxes may reserve on a host with this many CPUs.

    A quarter (at least one) stays free for gVisor, the sandbox server and the rest of the
    machine: with all 8 of 8 reserved, attempts ran about twice as slow and one timed out.
    """
    return max(1.0, cpus - max(1.0, cpus // 4))


class HostPool:
    def __init__(self, memory_mb: float, cpus: float):
        self.memory_mb, self.cpus = memory_mb, cpus
        self.used_memory = self.used_cpus = 0.0
        self.running: Counter = Counter()   # job id -> sandboxes running
        self.waiting: list = []             # [ticket, job id, memory, cpus]
        self._tickets = itertools.count()
        self._cond = threading.Condition()

    def _first(self) -> list:
        return min(self.waiting, key=lambda entry: (self.running[entry[1]], entry[0]))

    def _fits(self, memory_mb: float, cpus: float) -> bool:
        # a run bigger than the whole host still goes when nothing else runs: the host refuses it itself
        empty = not self.running
        return empty or (self.used_memory + memory_mb <= self.memory_mb and self.used_cpus + cpus <= self.cpus)

    def acquire(self, job_id, memory_mb: float, cpus: float,
                should_stop: Callable[[], Optional[str]] = lambda: None) -> None:
        entry = [next(self._tickets), job_id, memory_mb, cpus]
        with self._cond:
            self.waiting.append(entry)
        try:
            while True:
                with self._cond:
                    if self._first() is entry and self._fits(memory_mb, cpus):
                        self.waiting.remove(entry)
                        self.used_memory += memory_mb
                        self.used_cpus += cpus
                        self.running[job_id] += 1
                        self._cond.notify_all()   # the next in line may fit as well
                        return
                    self._cond.wait(WAIT_SECONDS)
                reason = should_stop()   # outside the lock: it may ask the database
                if reason:
                    raise RunStopped(reason)
        except BaseException:
            with self._cond:
                if entry in self.waiting:
                    self.waiting.remove(entry)
                    self._cond.notify_all()
            raise

    def release(self, job_id, memory_mb: float, cpus: float) -> None:
        with self._cond:
            self.used_memory -= memory_mb
            self.used_cpus -= cpus
            self.running[job_id] -= 1
            if self.running[job_id] <= 0:
                del self.running[job_id]
            self._cond.notify_all()

    @contextmanager
    def slot(self, job_id, memory_mb: float, cpus: float,
             should_stop: Callable[[], Optional[str]] = lambda: None):
        self.acquire(job_id, memory_mb, cpus, should_stop)
        try:
            yield
        finally:
            self.release(job_id, memory_mb, cpus)
