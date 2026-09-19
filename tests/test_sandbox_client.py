import io
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_gen_eval import sandbox_client
from code_gen_eval.sandbox_client import RETRY_DELAYS, SandboxClient, SandboxError, SandboxUnavailable


def client_for(handler):
    """A client against a fake sandbox API; returns it and the sleeps it asked for."""
    sleeps = []
    client = SandboxClient("http://host", "token", transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return client, sleeps


def replies(*answers):
    """A handler giving these answers in order: a status code, (status, json), or an exception class."""
    answers = list(answers)
    seen = []

    def handler(request):
        seen.append(request)
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, type) and issubclass(answer, Exception):
            raise answer("fake", request=request)
        status, body = answer if isinstance(answer, tuple) else (answer, {})
        headers = {"Retry-After": "3"} if status == 429 else {}
        return httpx.Response(status, json=body, headers=headers)

    return handler, seen


def archive(*entries) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def test_sends_the_token_and_reads_json():
    handler, seen = replies((201, {"id": "0f9e8d7c6b5a", "expires_at": 1}))
    client, _ = client_for(handler)
    assert client.create_sandbox(image="automl-runner", cpus=1) == "0f9e8d7c6b5a"
    assert seen[0].headers["Authorization"] == "Bearer token"


def test_waits_out_a_full_budget():
    handler, _ = replies(429, 429, (201, {"id": "abc"}))
    client, sleeps = client_for(handler)
    assert client.create_sandbox(image="automl-runner") == "abc"
    assert sleeps == [3.0, 3.0]


def test_gives_up_on_capacity_after_the_wait(monkeypatch):
    monkeypatch.setattr(sandbox_client, "CAPACITY_WAIT_SECONDS", 5)
    handler, _ = replies(429)
    client, sleeps = client_for(handler)
    with pytest.raises(SandboxUnavailable, match="no room"):
        client.create_sandbox(image="automl-runner")
    assert sleeps == [3.0, 3.0]


def test_rides_out_a_blip():
    handler, seen = replies(httpx.ConnectError, 502, (200, {"ok": True}))
    client, sleeps = client_for(handler)
    assert client.info() == {"ok": True}
    assert len(seen) == 3 and sleeps == list(RETRY_DELAYS[:2])


def test_an_unreachable_host_raises_after_the_backoff():
    handler, _ = replies(httpx.ConnectError)
    client, sleeps = client_for(handler)
    with pytest.raises(SandboxUnavailable, match="unreachable"):
        client.capacity()
    assert sleeps == list(RETRY_DELAYS)


def test_a_post_that_may_have_arrived_is_not_repeated():
    handler, seen = replies(httpx.ReadTimeout)
    client, sleeps = client_for(handler)
    with pytest.raises(SandboxUnavailable, match="stopped answering"):
        client.start_exec("sbx", ["python"], 60)
    assert len(seen) == 1 and sleeps == []


def test_a_refusal_is_not_retried():
    handler, seen = replies((409, {"detail": "sandbox 0f9e is exited, not running"}))
    client, _ = client_for(handler)
    with pytest.raises(SandboxError, match="not running") as error:
        client.put_files("sbx", {"a.py": b"x"})
    assert error.value.status == 409 and len(seen) == 1


def test_a_forgotten_exec_means_the_host_was_lost():
    handler, _ = replies((404, {"detail": "no exec"}))
    client, _ = client_for(handler)
    with pytest.raises(SandboxUnavailable):
        client.poll("sbx", "exec", 0, 0)


def test_deleting_what_is_gone_is_fine():
    handler, _ = replies((404, {"detail": "no sandbox"}))
    client, _ = client_for(handler)
    assert client.delete_sandbox("sbx") == {}
    client.delete_volume("vol")


def test_uploads_are_flat_tar_archives():
    handler, seen = replies((200, {"files": 2, "bytes": 3}))
    client, _ = client_for(handler)
    client.put_files("sbx", {"candidate.py": b"ab", "contract.json": b"c"})
    with tarfile.open(fileobj=io.BytesIO(seen[0].content)) as tar:
        assert sorted(tar.getnames()) == ["candidate.py", "contract.json"]


def test_download_extracts_into_the_attempt_directory(tmp_path):
    body = archive((".", None), ("./result.json", b"{}"), ("./catboost_info", None), ("./catboost_info/log.tsv", b"x"))
    client, _ = client_for(lambda request: httpx.Response(200, content=body))
    client.download("sbx", tmp_path)
    assert (tmp_path / "result.json").read_bytes() == b"{}"
    assert (tmp_path / "catboost_info" / "log.tsv").exists()


def test_download_refuses_an_escaping_archive(tmp_path):
    body = archive(("../escaped.txt", b"x"))
    client, _ = client_for(lambda request: httpx.Response(200, content=body))
    with pytest.raises(SandboxError, match="unsafe"):
        client.download("sbx", tmp_path / "attempt")
    assert not (tmp_path / "escaped.txt").exists()


def test_download_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_client, "MAX_DOWNLOAD_BYTES", 1000)
    body = archive(("./big.bin", b"x" * 5000))
    client, _ = client_for(lambda request: httpx.Response(200, content=body))
    with pytest.raises(SandboxError, match="over"):
        client.download("sbx", tmp_path)


def test_lists_a_jobs_sandboxes_by_label():
    handler, seen = replies((200, [{"id": "0f9e8d7c6b5a", "labels": {"run": "12"}}]))
    client, _ = client_for(handler)
    assert client.list_sandboxes({"run": "12", "model": "lightgbm"})[0]["id"] == "0f9e8d7c6b5a"
    assert seen[0].url.params.get_list("label") == ["run:12", "model:lightgbm"]


def test_each_retry_is_reported_to_on_wait():
    handler, _ = replies(httpx.ConnectError, httpx.ConnectError, (200, {"ok": True}))
    client, sleeps = client_for(handler)
    waits = []
    client.on_wait = lambda problem, delay: waits.append(delay)
    client.info()
    assert waits == sleeps == list(RETRY_DELAYS[:2])
