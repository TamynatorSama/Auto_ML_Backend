import docker.errors
import pytest

from conftest import EXEC, SANDBOX, TOKEN, VOLUME, make_tar

SANDBOX_BODY = {"image": "automl-runner", "cpus": 2, "memory_mb": 4096, "work_mb": 1024,
                "labels": {"job": "41", "model": "lightgbm"}}

ROUTES = [
    ("GET", "/v1/info"),
    ("GET", "/v1/capacity"),
    ("POST", "/v1/volumes"),
    ("PUT", f"/v1/volumes/{VOLUME}/files"),
    ("DELETE", f"/v1/volumes/{VOLUME}"),
    ("POST", "/v1/sandboxes"),
    ("GET", "/v1/sandboxes"),
    ("PUT", f"/v1/sandboxes/{SANDBOX}/files"),
    ("GET", f"/v1/sandboxes/{SANDBOX}/files?path=result.json"),
    ("POST", f"/v1/sandboxes/{SANDBOX}/kill"),
    ("DELETE", f"/v1/sandboxes/{SANDBOX}"),
    ("POST", f"/v1/sandboxes/{SANDBOX}/exec"),
    ("GET", f"/v1/sandboxes/{SANDBOX}/exec/{EXEC}"),
]


def test_health_needs_no_token(client):
    client.headers.pop("Authorization")
    assert client.get("/v1/health").json() == {"ok": True}


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_no_open_docs(client, path):
    client.headers.pop("Authorization")
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("method, path", ROUTES)
@pytest.mark.parametrize("header", [None, "Bearer wrong", TOKEN])
def test_every_route_needs_the_token(client, engine, method, path, header):
    client.headers.pop("Authorization")
    headers = {"Authorization": header} if header else {}
    assert client.request(method, path, headers=headers).status_code == 401
    assert engine.calls == []


def test_create_sandbox(client, engine, config):
    response = client.post("/v1/sandboxes", json=SANDBOX_BODY)
    assert response.status_code == 201
    assert response.json() == {"id": SANDBOX, "expires_at": 1}
    assert engine.calls == [("create_sandbox", "automl-runner", 2.0, 4096, 1024,
                             config.default_ttl_seconds, None, {"job": "41", "model": "lightgbm"})]


@pytest.mark.parametrize("change", [
    {"image": "python:3.12"},
    {"image": "automl-runner:current"},
    {"cpus": 4.5},
    {"cpus": 0},
    {"cpus": 1e-10},
    {"memory_mb": 6145},
    {"work_mb": 4097},
    {"ttl_seconds": 86401},
    {"volume": "../../etc"},
    {"labels": {"bad key": "x"}},
    {"labels": {"k": "v" * 257}},
    {"network": "host"},
])
def test_create_sandbox_refuses_bad_input(client, engine, change):
    response = client.post("/v1/sandboxes", json={**SANDBOX_BODY, **change})
    assert response.status_code == 400, response.text
    assert engine.calls == []


def test_no_capacity_is_429_with_retry_after(client, engine):
    engine.full = True
    response = client.post("/v1/sandboxes", json=SANDBOX_BODY)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert "8192" in response.json()["detail"]


def test_volume_create_and_delete(client, engine):
    assert client.post("/v1/volumes", json={"labels": {"job": "41"}, "ttl_seconds": 600}).status_code == 201
    assert client.delete(f"/v1/volumes/{VOLUME}").json() == {"deleted": VOLUME}
    assert engine.calls == [("create_volume", {"job": "41"}, 600), ("delete_volume", VOLUME)]


@pytest.mark.parametrize("path", [f"/v1/volumes/{VOLUME}/files", f"/v1/sandboxes/{SANDBOX}/files"])
def test_upload(client, engine, path):
    response = client.put(path, content=make_tar([("train.csv", b"a,b\n1,2\n"), ("folds.npz", b"xyz")]))
    assert response.json() == {"files": 2, "bytes": 11}
    assert len(engine.calls) == 1


@pytest.mark.parametrize("path", [f"/v1/volumes/{VOLUME}/files", f"/v1/sandboxes/{SANDBOX}/files"])
def test_upload_refuses_unsafe_and_large(client, engine, path):
    assert client.put(path, content=make_tar([("../x.py", b"x")])).status_code == 400
    assert client.put(path, content=b"x" * (1024 * 1024 + 1)).status_code == 413
    assert engine.calls == []


def test_unknown_ids(client):
    other = "ffffffffffff"
    assert client.put(f"/v1/volumes/{other}/files", content=make_tar([("a", b"")])).status_code == 404
    assert client.post(f"/v1/sandboxes/{other}/kill").status_code == 404
    assert client.delete(f"/v1/sandboxes/{other}").status_code == 404
    assert client.get(f"/v1/sandboxes/{SANDBOX}/exec/{other}").status_code == 404
    assert client.post(f"/v1/sandboxes/{other}/exec", json={"cmd": ["true"], "deadline_seconds": 5}).status_code == 404


def test_malformed_id_is_400(client):
    assert client.delete("/v1/sandboxes/not-an-id").status_code == 400


def test_download(client, engine):
    response = client.get(f"/v1/sandboxes/{SANDBOX}/files", params={"path": "./models/m.joblib"})
    assert response.content == b"tar-bytes"
    assert response.headers["content-type"] == "application/x-tar"
    assert engine.calls == [("get_files", SANDBOX, "models/m.joblib")]


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd", "a/../../b"])
def test_download_stays_inside_work(client, path):
    assert client.get(f"/v1/sandboxes/{SANDBOX}/files", params={"path": path}).status_code == 400


def test_exec(client, engine):
    body = {"cmd": ["python", "-m", "automl_runtime.evaluate"], "deadline_seconds": 405}
    response = client.post(f"/v1/sandboxes/{SANDBOX}/exec", json=body)
    assert (response.status_code, response.json()) == (202, {"exec_id": EXEC})
    poll = client.get(f"/v1/sandboxes/{SANDBOX}/exec/{EXEC}", params={"stdout_offset": 12})
    assert poll.json() == {"state": "running", "stdout_offset": 12}


@pytest.mark.parametrize("body", [
    {"cmd": ["true"], "deadline_seconds": 7201},
    {"cmd": [], "deadline_seconds": 5},
    {"cmd": "rm -rf /", "deadline_seconds": 5},
])
def test_exec_refuses_bad_input(client, engine, body):
    assert client.post(f"/v1/sandboxes/{SANDBOX}/exec", json=body).status_code == 400
    assert engine.calls == []


def test_kill_and_delete(client):
    assert client.post(f"/v1/sandboxes/{SANDBOX}/kill").json() == {"killed": SANDBOX}
    assert client.delete(f"/v1/sandboxes/{SANDBOX}").json() == {"deleted": SANDBOX, "usage": {"wall_seconds": 1.0}}


def test_list_label_filters(client, engine):
    assert client.get("/v1/sandboxes?label=job:41&label=model:light:gbm").json() == []
    assert engine.calls == [("list_sandboxes", [("job", "41"), ("model", "light:gbm")])]
    assert client.get("/v1/sandboxes?label=job").status_code == 400


class FakeResponse:
    def __init__(self, status):
        self.status_code = status
        self.reason = "x"


@pytest.mark.parametrize("error, status", [
    (docker.errors.APIError("in use", FakeResponse(409), "volume is in use"), 409),
    (docker.errors.NotFound("gone", FakeResponse(404), "no such container"), 404),
    (docker.errors.APIError("boom", FakeResponse(500), "daemon exploded"), 502),
    (docker.errors.DockerException("socket unreachable"), 502),
])
def test_docker_errors(client, engine, error, status):
    engine.error = error
    response = client.get("/v1/info")
    assert response.status_code == status
    assert response.json()["detail"].startswith("docker: ")
