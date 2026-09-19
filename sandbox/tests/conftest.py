import io
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl_sandbox.app import create_app  # noqa: E402
from automl_sandbox.config import from_dict  # noqa: E402
from automl_sandbox.errors import NoCapacity, NotFound  # noqa: E402

TOKEN = "t" * 40
SANDBOX = "0f9e8d7c6b5a"
VOLUME = "a1b2c3d4e5f6"
EXEC = "5e4d3c2b1a09"


class FakeEngine:
    """Stands in for Docker: one known volume, one sandbox and one exec."""

    def __init__(self):
        self.calls = []
        self.full = False
        self.error = None  # raised by every call when set

    def _record(self, name, *args):
        if self.error:
            raise self.error
        self.calls.append((name, *args))

    def _sandbox(self, sid):
        if sid != SANDBOX:
            raise NotFound(f"no sandbox {sid}")

    def info(self):
        self._record("info")
        return {"api_version": "1"}

    def capacity(self):
        return {"cpus": 8, "memory_budget_mb": 8192, "memory_reserved_mb": 0, "memory_free_mb": 8192, "sandboxes_running": 0}

    def create_volume(self, labels, ttl):
        self._record("create_volume", labels, ttl)
        return {"id": VOLUME, "expires_at": 1}

    def put_volume_files(self, vid, archive):
        if vid != VOLUME:
            raise NotFound(f"no volume {vid}")
        self._record("put_volume_files", vid, archive)

    def delete_volume(self, vid):
        self._record("delete_volume", vid)

    def create_sandbox(self, image, cpus, memory_mb, work_mb, ttl, volume, labels):
        if self.full:
            raise NoCapacity("memory budget: 8192 of 8192 MB reserved, 1024 requested")
        self._record("create_sandbox", image, cpus, memory_mb, work_mb, ttl, volume, labels)
        return {"id": SANDBOX, "expires_at": 1}

    def list_sandboxes(self, labels):
        self._record("list_sandboxes", labels)
        return []

    def put_files(self, sid, archive):
        self._sandbox(sid)
        self._record("put_files", sid, archive)

    def get_files(self, sid, path):
        self._sandbox(sid)
        self._record("get_files", sid, path)
        return iter([b"tar-", b"bytes"])

    def kill(self, sid):
        self._sandbox(sid)

    def delete_sandbox(self, sid):
        self._sandbox(sid)
        return {"wall_seconds": 1.0}

    def start_exec(self, sid, cmd, deadline):
        self._sandbox(sid)
        self._record("start_exec", sid, cmd, deadline)
        return EXEC

    def get_exec(self, sid, exec_id, stdout_offset, stderr_offset):
        self._sandbox(sid)
        if exec_id != EXEC:
            raise NotFound(f"no exec {exec_id}")
        return {"state": "running", "stdout_offset": stdout_offset}


def make_tar(entries) -> bytes:
    """entries: TarInfo objects, or (name, bytes) pairs for plain files."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for entry in entries:
            if isinstance(entry, tarfile.TarInfo):
                tar.addfile(entry)
            else:
                name, data = entry
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


@pytest.fixture
def config():
    return from_dict({"token": TOKEN, "max_upload_mb": 1})


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(config, engine):
    return TestClient(create_app(config, engine), headers={"Authorization": f"Bearer {TOKEN}"})
