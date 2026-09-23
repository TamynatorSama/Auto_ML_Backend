from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_gen_eval import runner


class FakeSandboxClient:
    def __init__(self):
        self.downloads = 0
        self.deleted = []

    def create_sandbox(self, **spec):
        return "sandbox-id"

    def put_files(self, sandbox, files):
        pass

    def start_exec(self, sandbox, command, deadline):
        return "exec-id"

    def download(self, sandbox, out_dir):
        self.downloads += 1

    def delete_sandbox(self, sandbox, retry=True):
        self.deleted.append(sandbox)
        return {"peak_memory_mb": 1600.0, "oom_kills": 1}


@pytest.mark.parametrize(("oom_killed", "expected_downloads"), [(False, 1), (True, 0)])
def test_sandbox_oom_does_not_download_from_stopped_container(
    monkeypatch, oom_killed, expected_downloads
):
    client = FakeSandboxClient()
    context = SimpleNamespace(run_id=41, n_jobs=1, reserve_memory_mb=512, reserve_cpus=1)
    status = {
        "state": "exited",
        "exit_code": 137 if oom_killed else 0,
        "elapsed_seconds": 3.2,
        "oom_killed": oom_killed,
    }
    emitted = []

    monkeypatch.setattr(runner.hooks, "sandbox_for", lambda run_id: client)
    monkeypatch.setattr(runner.hooks, "sandbox_slot", lambda *args: nullcontext())
    monkeypatch.setattr(runner.hooks, "emit", lambda *args, **kwargs: emitted.append((args, kwargs)))
    monkeypatch.setattr(runner, "_run_volume", lambda *args, **kwargs: "volume-id")
    monkeypatch.setattr(runner, "_follow", lambda *args: (status, "stdout", "stderr", False))

    out_dir = Path("unused") / "mlp" / "attempt_1"
    result = runner._sandbox_execute(
        context, out_dir, ["python", "evaluate.py"], {}, "training", 120, 1638, "loop"
    )

    assert client.downloads == expected_downloads
    assert client.deleted == ["sandbox-id"]
    assert result[7] is oom_killed
    assert result[8]["oom_kills"] == 1
    assert emitted[0][1]["model"] == "mlp"
