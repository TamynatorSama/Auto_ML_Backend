import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_gen_eval import runner
from code_gen_eval.sandbox_client import SandboxError, SandboxUnavailable
from test_values import context as fixture_context
from utils.reusable import hooks
from utils.reusable.environment import probe_environment, runtime_hash
from utils.reusable.resources import plan_resources

ROOT = Path(__file__).resolve().parent.parent

MODULE = """from sklearn.linear_model import Ridge


def build_pipeline(columns, task, ctx):
    return Ridge(alpha=1.0)
"""

TRACEBACK = 'Traceback (most recent call last):\n  File "x.py"\nValueError: bad input\n'


def status(state="running", err="", exit_code=None, oom=False):
    return {
        "state": state, "exit_code": exit_code, "oom_killed": oom, "elapsed_seconds": 2.5,
        "stdout": "", "stdout_offset": 0, "stdout_skipped": False,
        "stderr": err, "stderr_offset": 0, "stderr_skipped": False, "live": None,
    }


class FakeHost:
    """A sandbox host in memory. Each sandbox plays the next (polls, files) script."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.sandboxes = []
        self.volumes = {}
        self.deleted_volumes = []
        self.missing_volumes = set()

    def create_volume(self, labels, ttl):
        volume = f"vol{len(self.volumes)}"
        self.volumes[volume] = {"labels": labels}
        return volume

    def put_volume_files(self, volume, files):
        self.volumes[volume]["files"] = sorted(files)

    def delete_volume(self, volume, retry=True):
        self.deleted_volumes.append(volume)

    def create_sandbox(self, **spec):
        if spec["volume"] in self.missing_volumes:
            raise SandboxError(404, f"no volume {spec['volume']}")
        polls, files = self.scripts.pop(0)
        self.sandboxes.append({"spec": spec, "polls": list(polls), "files": files, "deleted": False})
        return len(self.sandboxes) - 1

    def put_files(self, sandbox, files):
        self.sandboxes[sandbox]["uploaded"] = sorted(files)

    def start_exec(self, sandbox, cmd, deadline):
        self.sandboxes[sandbox].update(cmd=cmd, deadline=deadline)
        return "exec"

    def poll(self, sandbox, exec_id, stdout_offset, stderr_offset):
        polls = self.sandboxes[sandbox]["polls"]
        return polls.pop(0) if len(polls) > 1 else polls[0]

    def kill(self, sandbox):
        self.sandboxes[sandbox]["polls"] = [status("killed")]

    def download(self, sandbox, dest, path="."):
        for name, content in self.sandboxes[sandbox]["files"].items():
            (dest / name).write_text(content if isinstance(content, str) else json.dumps(content))

    def delete_sandbox(self, sandbox, retry=True):
        self.sandboxes[sandbox]["deleted"] = True
        return {"cpu_seconds": 4.2, "peak_memory_mb": 181.2, "measured_with": "cgroup-v2"}


@pytest.fixture
def run(tmp_path, monkeypatch):
    context = fixture_context.model_copy(
        update={"backend": "sandbox", "run_dir": str(tmp_path / "run"), "worker_memory_mb": 2048.0}
    )
    monkeypatch.setattr(runner, "SANDBOX_POLL_SECONDS", 0)

    def start(*scripts):
        host = FakeHost(*scripts)
        monkeypatch.setattr(hooks, "sandbox_for", lambda run_id: host)
        return host

    return context, start


def ok_result(**extra):
    return {"status": "ok", "stage": "", "error": "", "cv_scores": {fixture_context.primary_metric: 0.5}, **extra}


def test_a_loop_attempt_runs_in_a_sandbox_without_the_test_set(run):
    context, start = run
    host = start(([status(), status("exited", exit_code=0)], {"result.json": ok_result()}))

    record = runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")

    assert record.status == "ok" and record.cv_scores
    assert record.usage["cpu_seconds"] == 4.2 and record.peak_memory_mb == 181.2 and record.wall_seconds == 2.5
    sandbox = host.sandboxes[0]
    assert sandbox["deleted"] and sandbox["uploaded"] == ["candidate.py", "contract.json"]
    assert sandbox["cmd"] == ["python", "-m", "automl_runtime.evaluate", "--attempt-dir", "/work"]
    assert sandbox["spec"]["memory_mb"] == 2048 and sandbox["spec"]["labels"]["step"] == "loop"
    (volume,) = host.volumes.values()
    assert Path(context.test_path).name not in volume["files"]

    contract = json.loads((Path(context.run_dir) / "ridge" / "attempt_1" / "contract.json").read_text())
    assert contract["train_path"].startswith("/data/") and contract["folds_path"].startswith("/data/")


def test_the_final_run_and_the_load_check_each_get_a_fresh_sandbox_with_the_test_set(run):
    context, start = run
    final = ok_result(test_scores={context.primary_metric: 0.4}, artifacts={"model": "model.joblib"})
    host = start(
        ([status("exited", exit_code=0)], {"result_final.json": final, "model.joblib": "pickle"}),
        ([status("exited", exit_code=0)], {"load_check.json": {"status": "ok", "error": ""}}),
    )

    record = runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox", final=True)

    assert record.test_scores and not any("cannot be loaded" in w for w in record.warnings)
    fitted, checked = host.sandboxes
    assert "--final" in fitted["cmd"] and fitted["cmd"][-1] == f"/data/{Path(context.test_path).name}"
    assert "--load-check" in checked["cmd"] and checked["spec"]["labels"]["step"] == "load_check"
    assert checked["uploaded"] == ["candidate.py", "contract.json", "model.joblib"]
    assert fitted["spec"]["volume"] == checked["spec"]["volume"]
    assert Path(context.test_path).name in host.volumes[fitted["spec"]["volume"]]["files"]


@pytest.mark.parametrize("poll, expected", [
    (status("exited", exit_code=128, oom=True), "out_of_memory"),
    (status("timeout"), "timeout"),
    (status("exited", exit_code=1, err=TRACEBACK), "error"),
])
def test_how_an_attempt_ends(run, poll, expected):
    context, start = run
    host = start(([poll], {}))
    record = runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    assert record.status == expected
    assert host.sandboxes[0]["deleted"]


def test_a_traceback_that_never_exits_kills_the_sandbox(run, monkeypatch):
    context, start = run
    monkeypatch.setattr(runner, "ERROR_GRACE_SECONDS", -1)
    host = start(([status(err=TRACEBACK)], {}))
    record = runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    assert record.status == "error" and "did not exit" in record.traceback
    assert host.sandboxes[0]["deleted"]


def test_a_lost_exec_stops_the_model_instead_of_failing_the_attempt(run):
    context, start = run
    host = start(([status("lost")], {}))
    with pytest.raises(SandboxUnavailable):
        runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    assert host.sandboxes[0]["deleted"]
    assert not (Path(context.run_dir) / "ridge" / "attempt_1" / "record.json").exists()


def test_expired_run_data_is_uploaded_again(run):
    context, start = run
    host = start(
        ([status("exited", exit_code=0)], {"result.json": ok_result()}),
        ([status("exited", exit_code=0)], {"result.json": ok_result()}),
    )
    runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    host.missing_volumes.add(host.sandboxes[0]["spec"]["volume"])
    runner.run_candidate(MODULE + "\n", "ridge", 2, context, backend="sandbox")
    assert len(host.volumes) == 2 and host.sandboxes[1]["spec"]["volume"] == "vol1"


def slots_held(monkeypatch, host):
    """Records sandbox_slot: what each asked for, and whether its sandbox was gone when it was given back."""
    held = []

    @contextmanager
    def slot(run_id, memory_mb, cpus):
        held.append({"memory_mb": memory_mb, "cpus": cpus})
        try:
            yield
        finally:
            held[-1]["deleted_before_release"] = all(sandbox["deleted"] for sandbox in host.sandboxes)

    monkeypatch.setattr(hooks, "sandbox_slot", slot)
    return held


def test_each_sandbox_holds_a_slot_from_create_to_delete(run, monkeypatch):
    context, start = run
    host = start(([status("exited", exit_code=0)], {"result.json": ok_result()}))
    held = slots_held(monkeypatch, host)
    runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    assert held == [{"memory_mb": 2048, "cpus": context.n_jobs, "deleted_before_release": True}]


def test_a_lost_sandbox_gives_its_slot_back(run, monkeypatch):
    context, start = run
    host = start(([status("lost")], {}))
    held = slots_held(monkeypatch, host)
    with pytest.raises(SandboxUnavailable):
        runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    assert held[0]["deleted_before_release"]


def test_the_run_releases_its_data(run):
    context, start = run
    host = start(([status("exited", exit_code=0)], {"result.json": ok_result()}))
    runner.run_candidate(MODULE, "ridge", 1, context, backend="sandbox")
    runner.release_run(context)
    assert host.deleted_volumes == ["vol0"]


# ---- environment and capacity from the host


class InfoHost:
    def __init__(self, runtime):
        self.runtime = runtime

    def info(self):
        packages = {"scikit-learn": "1.9.1", "pandas": "3.0.3", "numpy": "2.5.3", "lightgbm": "4.7.0"}
        return {
            "images": {"automl-runner": {"runtime_hash": self.runtime, "packages": packages}},
            "limits": {"max_sandbox_memory_mb": 6144, "max_cpus": 4.0},
        }

    def capacity(self):
        # busy with another job's sandboxes: the plan still comes from the whole budget
        return {"cpus": 8, "memory_budget_mb": 8192, "memory_free_mb": 1024}


def test_runtime_hash_matches_the_script_that_labels_the_image():
    spec = importlib.util.spec_from_file_location("runtime_hash", ROOT / "sandbox" / "scripts" / "runtime_hash.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.runtime_hash() == runtime_hash()


def test_environment_comes_from_the_host_image():
    found = probe_environment("sandbox", InfoHost(runtime_hash()))
    assert found == {"sklearn": "1.9.1", "pandas": "3.0.3", "numpy": "2.5.3", "lightgbm": "4.7.0"}


def test_a_host_with_a_different_runtime_is_refused():
    with pytest.raises(RuntimeError, match="rebuild its runner image"):
        probe_environment("sandbox", InfoHost("0000000000000000"))


def test_the_plan_fits_the_host_budget(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("a,b\n" + "1,2\n" * 1000)
    plan = plan_resources(str(train), 5, InfoHost(runtime_hash()))
    assert plan.max_concurrency == 4 and plan.n_jobs == 2
    assert plan.worker_memory_mb * plan.max_concurrency <= 8192
    assert plan.worker_memory_mb <= 6144
