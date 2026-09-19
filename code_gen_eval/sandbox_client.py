"""
sandbox_client.py
-----------------
Talks to a sandbox server (sandbox/DESIGN.md). One client per host; which host a
run uses is the caller's choice, through hooks.sandbox_for.
"""

from __future__ import annotations

import io
import os
import tarfile
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import httpx

RETRY_DELAYS = (5, 10, 20, 40, 60, 60, 60, 60)  # about five minutes in all
CAPACITY_WAIT_SECONDS = 1800                     # how long 429s are waited out
MAX_DOWNLOAD_BYTES = 1 << 30
UPLOAD_TIMEOUT = 600                             # a 100 MB file through a slow tunnel
# a POST that may have reached the server is not repeated: a second exec would run the
# evaluator twice in one sandbox. These errors mean the request never arrived.
NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout)


class SandboxUnavailable(Exception):
    """The host could not be reached, or had no room, for longer than we wait."""


class SandboxError(Exception):
    """The host answered and refused."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"sandbox host answered {status}: {detail}")
        self.status = status


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", ""))[:500]
    except ValueError:
        return response.text[:500]


def tar_files(files: Dict[str, Union[bytes, Path]]) -> bytes:
    """A flat archive of name -> content (bytes, or a file to read)."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, source in files.items():
            data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


class SandboxClient:
    def __init__(self, url: str, token: str, transport: Optional[httpx.BaseTransport] = None, sleep=time.sleep,
                 on_wait: Optional[Callable[[str, float], None]] = None):
        self.url = url.rstrip("/")
        self.http = httpx.Client(
            base_url=self.url, headers={"Authorization": f"Bearer {token}"}, timeout=60, transport=transport,
            # drop idle connections before the server's 5 s keep-alive does, or a request can
            # go out on a connection the server is closing, and a POST can't be retried
            limits=httpx.Limits(keepalive_expiry=2),
        )
        self.sleep = sleep
        self.on_wait = on_wait  # (problem, delay) before each retry of an unreachable host

    @classmethod
    def from_env(cls) -> "SandboxClient":
        token = os.environ.get("AUTOML_SANDBOX_TOKEN")
        if not token:
            raise RuntimeError("AUTOML_SANDBOX_TOKEN is not set: the sandbox backend needs the server's token")
        return cls(os.environ.get("AUTOML_SANDBOX_URL", "http://127.0.0.1:8765"), token)

    # ---- transport

    def _request(self, method: str, path: str, retry: bool = True, **kwargs) -> httpx.Response:
        """A streaming response for any 2xx; waits out 429s and retries an unreachable host."""
        delays = iter(RETRY_DELAYS if retry else ())
        waited = 0.0
        while True:
            try:
                response = self.http.send(self.http.build_request(method, path, **kwargs), stream=True)
            except httpx.TransportError as error:
                if method == "POST" and not isinstance(error, NEVER_SENT):
                    raise SandboxUnavailable(f"sandbox host stopped answering during {method} {path}: {error!r}")
                problem = repr(error)
            else:
                if response.status_code < 400:
                    return response
                response.read()
                response.close()
                if response.status_code == 429:
                    if waited >= CAPACITY_WAIT_SECONDS:
                        raise SandboxUnavailable(f"no room on the sandbox host for {waited / 60:.0f} min: {_detail(response)}")
                    delay = float(response.headers.get("Retry-After") or 5)
                    self.sleep(delay)
                    waited += delay
                    continue
                if response.status_code < 500:
                    raise SandboxError(response.status_code, _detail(response))
                problem = f"{response.status_code} {_detail(response)}"

            delay = next(delays, None)
            if delay is None:
                raise SandboxUnavailable(f"sandbox host unreachable at {self.url}: {problem}")
            print(f"  sandbox host: {problem}; retrying in {delay}s", flush=True)
            if self.on_wait:
                self.on_wait(problem, delay)
            self.sleep(delay)

    def _json(self, method: str, path: str, retry: bool = True, **kwargs):
        response = self._request(method, path, retry, **kwargs)
        try:
            response.read()
            return response.json() if response.content else {}
        finally:
            response.close()

    # ---- host

    def info(self, retry: bool = True) -> dict:
        return self._json("GET", "/v1/info", retry)

    def capacity(self) -> dict:
        return self._json("GET", "/v1/capacity")

    # ---- run volumes

    def create_volume(self, labels: dict, ttl_seconds: int) -> str:
        return self._json("POST", "/v1/volumes", json={"labels": labels, "ttl_seconds": ttl_seconds})["id"]

    def put_volume_files(self, volume: str, files: Dict[str, Union[bytes, Path]]) -> None:
        self._json("PUT", f"/v1/volumes/{volume}/files", content=tar_files(files), timeout=UPLOAD_TIMEOUT)

    def delete_volume(self, volume: str, retry: bool = True) -> None:
        try:
            self._json("DELETE", f"/v1/volumes/{volume}", retry)
        except SandboxError as error:
            if error.status not in (404, 409):  # gone already, or the reaper will take it
                raise

    # ---- sandboxes

    def list_sandboxes(self, labels: dict) -> list:
        params = [("label", f"{key}:{value}") for key, value in labels.items()]
        return self._json("GET", "/v1/sandboxes", params=params)

    def create_sandbox(self, **spec) -> str:
        return self._json("POST", "/v1/sandboxes", json=spec)["id"]

    def put_files(self, sandbox: str, files: Dict[str, Union[bytes, Path]]) -> None:
        self._json("PUT", f"/v1/sandboxes/{sandbox}/files", content=tar_files(files), timeout=UPLOAD_TIMEOUT)

    def start_exec(self, sandbox: str, cmd: list, deadline_seconds: float) -> str:
        body = {"cmd": cmd, "deadline_seconds": deadline_seconds}
        return self._json("POST", f"/v1/sandboxes/{sandbox}/exec", json=body)["exec_id"]

    def poll(self, sandbox: str, exec_id: str, stdout_offset: int, stderr_offset: int) -> dict:
        params = {"stdout_offset": stdout_offset, "stderr_offset": stderr_offset}
        try:
            return self._json("GET", f"/v1/sandboxes/{sandbox}/exec/{exec_id}", params=params)
        except SandboxError as error:
            if error.status == 404:  # the server restarted and forgot the exec
                raise SandboxUnavailable(f"the sandbox host lost track of the run: {error}")
            raise

    def kill(self, sandbox: str) -> None:
        self._json("POST", f"/v1/sandboxes/{sandbox}/kill")

    def delete_sandbox(self, sandbox: str, retry: bool = True) -> dict:
        try:
            return self._json("DELETE", f"/v1/sandboxes/{sandbox}", retry).get("usage") or {}
        except SandboxError as error:
            if error.status == 404:
                return {}
            raise

    def download(self, sandbox: str, dest: Path, path: str = ".") -> None:
        """Everything under /work/<path>, extracted into dest."""
        response = self._request("GET", f"/v1/sandboxes/{sandbox}/files", params={"path": path}, timeout=UPLOAD_TIMEOUT)
        data = bytearray()
        try:
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise SandboxError(413, f"the attempt's files are over {MAX_DOWNLOAD_BYTES >> 20} MB")
        except httpx.TransportError as error:
            raise SandboxUnavailable(f"download from the sandbox host broke off: {error!r}")
        finally:
            response.close()

        try:
            with tarfile.open(fileobj=io.BytesIO(bytes(data))) as tar:
                tar.extractall(dest, filter="data")
        except tarfile.TarError as error:
            raise SandboxError(502, f"unsafe or unreadable archive from the sandbox host: {error}")
