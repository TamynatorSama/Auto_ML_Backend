import threading
import time

import pytest

from utils.reusable.hooks import RunStopped
from worker import pool as pool_module
from worker.pool import HostPool, usable_cpus


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(pool_module, "WAIT_SECONDS", 0.02)


class Line:
    """Runs that ask for places from their own threads, recording the order they start in."""

    def __init__(self, host: HostPool):
        self.host, self.started, self.errors = host, [], []

    def seen(self):
        return len(self.host.waiting) + len(self.started) + len(self.errors)

    def ask(self, name, job, memory, cpus, should_stop=lambda: None):
        def run():
            try:
                self.host.acquire(job, memory, cpus, should_stop)
                self.started.append(name)
            except BaseException as error:
                self.errors.append((name, error))

        before = self.seen()
        threading.Thread(target=run, daemon=True).start()
        while self.seen() == before:   # in line (or started) before the next one asks
            time.sleep(0.005)

    def settle(self):
        time.sleep(0.15)


def test_the_job_with_fewer_running_goes_first():
    host = HostPool(memory_mb=4096, cpus=8)
    host.acquire("a", 2048, 2)
    host.acquire("a", 2048, 2)          # job a holds the whole host
    line = Line(host)
    line.ask("a-3", "a", 2048, 2)
    line.ask("b-1", "b", 2048, 2)       # asked later, but job b has nothing running
    line.settle()
    assert line.started == []

    host.release("a", 2048, 2)
    line.settle()
    assert line.started == ["b-1"]
    host.release("a", 2048, 2)
    line.settle()
    assert line.started == ["b-1", "a-3"]


def test_runs_of_one_job_go_in_the_order_asked():
    host = HostPool(memory_mb=2048, cpus=8)
    host.acquire("a", 2048, 2)
    line = Line(host)
    line.ask("a-2", "a", 2048, 2)
    line.ask("a-3", "a", 2048, 2)
    host.release("a", 2048, 2)
    line.settle()
    assert line.started == ["a-2"]


def test_a_run_waits_for_cpus_as_well_as_memory():
    host = HostPool(memory_mb=8192, cpus=4)
    host.acquire("a", 1024, 4)
    line = Line(host)
    line.ask("b-1", "b", 1024, 2)       # plenty of memory, no cpus
    line.settle()
    assert line.started == []
    host.release("a", 1024, 4)
    line.settle()
    assert line.started == ["b-1"]


def test_the_first_in_line_is_not_skipped_for_a_smaller_run():
    host = HostPool(memory_mb=4096, cpus=8)
    host.acquire("a", 3072, 2)
    line = Line(host)
    line.ask("b-big", "b", 2048, 2)     # first in line, doesn't fit yet
    line.ask("c-small", "c", 512, 2)    # would fit, but waits its turn
    line.settle()
    assert line.started == []
    host.release("a", 3072, 2)
    line.settle()
    assert line.started == ["b-big", "c-small"]


def test_a_stop_while_waiting_leaves_the_line():
    host = HostPool(memory_mb=2048, cpus=8)
    host.acquire("a", 2048, 2)
    stop = {"reason": None}
    line = Line(host)
    line.ask("b-1", "b", 2048, 2, lambda: stop["reason"])
    stop["reason"] = "stop requested"
    line.settle()
    assert [(name, type(error)) for name, error in line.errors] == [("b-1", RunStopped)]
    assert host.waiting == []


def test_a_place_is_given_back_when_the_run_fails():
    host = HostPool(memory_mb=2048, cpus=8)
    with pytest.raises(RuntimeError):
        with host.slot("a", 2048, 2):
            raise RuntimeError("the sandbox host dropped")
    assert (host.used_memory, host.used_cpus, dict(host.running)) == (0, 0, {})


def test_a_run_bigger_than_the_host_still_goes_when_it_is_empty():
    host = HostPool(memory_mb=2048, cpus=2)
    with host.slot("a", 6144, 4):   # the host itself refuses what it can't hold
        assert host.running["a"] == 1


def test_a_quarter_of_the_cpus_stays_free_for_the_machine():
    assert [usable_cpus(n) for n in (1, 2, 4, 8, 16)] == [1, 1, 3, 6, 12]
