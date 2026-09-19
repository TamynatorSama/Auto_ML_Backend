"""End-to-end test of a running sandbox server, standard library only. Run it on the server:

    python3 sandbox/scripts/smoke.py --replay /tmp/automl-spike/spike/replay

Prints a SMOKE SUMMARY block to paste back.
"""

import argparse
import io
import json
import subprocess
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path

NET_PROBE = """
import socket
for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53), ("172.17.0.1", 22)):
    try:
        socket.create_connection((host, port), timeout=3).close()
        print(f"{host}:{port} REACHABLE")
    except OSError:
        print(f"{host}:{port} blocked")
"""
FS_PROBE = """
import os
for path in ("/", "/etc", "/opt/automl", "/data", "/work", "/tmp"):
    try:
        open(os.path.join(path, ".probe"), "w").close()
        os.remove(os.path.join(path, ".probe"))
        print(path, "writable")
    except OSError:
        print(path, "refused")
"""
OOM_PROBE = """
chunks = []
for i in range(64):
    block = bytearray(256 * 1024 * 1024)
    block[::4096] = b"\\x01" * (len(block) // 4096)
    chunks.append(block)
    print(f"allocated {(i + 1) * 256} MB", flush=True)
"""


class Api:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def call(self, method, path, body=None, raw=None, token=True):
        data = json.dumps(body).encode() if body is not None else raw
        headers = {"Content-Type": "application/json" if body is not None else "application/x-tar"}
        if token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def json(self, method, path, body=None, raw=None):
        status, _, data = self.call(method, path, body, raw)
        if status >= 300:
            raise RuntimeError(f"{method} {path} -> {status}: {data[:300].decode(errors='replace')}")
        return json.loads(data)


def tar_of(files: dict) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def first_file(archive: bytes) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        return tar.extractfile(next(m for m in tar if m.isfile())).read()


def run(api, sid, cmd, deadline):
    """Start an exec and poll it to the end -> final status with the whole output."""
    exec_id = api.json("POST", f"/v1/sandboxes/{sid}/exec", {"cmd": cmd, "deadline_seconds": deadline})["exec_id"]
    out = err = ""
    out_at = err_at = 0
    while True:
        status = api.json("GET", f"/v1/sandboxes/{sid}/exec/{exec_id}?stdout_offset={out_at}&stderr_offset={err_at}")
        out, err = out + status["stdout"], err + status["stderr"]
        out_at, err_at = status["stdout_offset"], status["stderr_offset"]
        if status["state"] != "running":
            status.update(stdout=out, stderr=err)
            return status
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--config", default="/etc/automl-sandbox/config.toml")
    parser.add_argument("--replay", required=True, help="the spike's replay folder: data/, work/, expected.json")
    args = parser.parse_args()

    with open(args.config, "rb") as handle:
        api = Api(args.url, tomllib.load(handle)["token"])
    replay = Path(args.replay)
    expected = json.loads((replay / "expected.json").read_text())
    tag = {"smoke": uuid.uuid4().hex[:8]}
    results, notes = [], []

    def check(name, ok, detail):
        results.append(f"{'PASS' if ok else 'FAIL'}  {name:<13} {detail}")
        print(results[-1], flush=True)

    def sandbox(memory_mb, **extra):
        body = {"image": "automl-runner", "cpus": 2, "memory_mb": memory_mb, "work_mb": 256,
                "ttl_seconds": 1800, "labels": tag, **extra}
        return api.json("POST", "/v1/sandboxes", body)["id"]

    def attempt(name, test):
        try:
            test()
        except Exception as error:
            check(name, False, f"{type(error).__name__}: {error}")

    # ---- auth and info
    def auth():
        health, _, _ = api.call("GET", "/v1/health", token=False)
        bare, _, _ = api.call("GET", "/v1/info", token=False)
        check("auth", health == 200 and bare == 401, f"health {health} without token, info {bare} without token")

    def info():
        info = api.json("GET", "/v1/info")
        runner = info["images"].get("automl-runner", {})
        notes.append(f"server:  {info['runtime']} / {info['isolation']}, measured with {info['measurement']}, "
                     f"runner hash {runner.get('runtime_hash')}")
        packages = runner.get("packages", {})
        notes.append("image:   " + ", ".join(f"{p} {packages.get(p)}" for p in ("scikit-learn", "pandas", "numpy", "lightgbm")))
        check("info", bool(runner.get("runtime_hash")), f"runner image {runner.get('tag')}, error {runner.get('error')}")

    # ---- the replay, then probes inside the same sandbox
    def replay_attempt():
        volume = api.json("POST", "/v1/volumes", {"labels": tag, "ttl_seconds": 3600})["id"]
        try:
            data = {p.name: p.read_bytes() for p in (replay / "data").iterdir() if p.is_file()}
            api.json("PUT", f"/v1/volumes/{volume}/files", raw=tar_of(data))
            sid = sandbox(4096, volume=volume, work_mb=1024)
            work = {p.name: p.read_bytes() for p in (replay / "work").iterdir() if p.is_file()}
            api.json("PUT", f"/v1/sandboxes/{sid}/files", raw=tar_of(work))

            status = run(api, sid, ["python", "-m", "automl_runtime.evaluate", "--attempt-dir", "/work"], 600)
            metric = expected["primary_metric"]
            original = expected["cv_scores"][metric]
            _, _, archive = api.call("GET", f"/v1/sandboxes/{sid}/files?path=result.json")
            try:
                value = json.loads(first_file(archive)).get("cv_scores", {}).get(metric)
            except Exception:
                value = None
            if value is None:
                check("replay", False, f"state {status['state']}, exit {status['exit_code']}: {status['stderr'][-300:]}")
            else:
                same = abs(value - original) < 0.005
                check("replay", same, f"{metric} {value:.4f} vs original {original:.4f} "
                      f"({'MATCH' if same else 'DIFFERENT'}) in {status['elapsed_seconds']} s")

            net = run(api, sid, ["python", "-c", NET_PROBE], 30)["stdout"]
            check("network", "REACHABLE" not in net and "blocked" in net, net.strip().replace("\n", "; "))
            fs = run(api, sid, ["python", "-c", FS_PROBE], 30)["stdout"]
            wanted = {"/": "refused", "/etc": "refused", "/opt/automl": "refused", "/data": "refused",
                      "/work": "writable", "/tmp": "writable"}
            got = dict(line.split() for line in fs.split("\n") if line.strip())
            check("filesystem", got == wanted, fs.strip().replace("\n", "; "))

            busy, _, _ = api.call("DELETE", f"/v1/volumes/{volume}")
            check("volume busy", busy == 409, f"delete while mounted -> {busy}")
            usage = api.json("DELETE", f"/v1/sandboxes/{sid}")["usage"]
            notes.append(f"usage:   wall {usage['wall_seconds']} s, cpu {usage['cpu_seconds']} s, "
                         f"peak {usage['peak_memory_mb']} MB, oom {usage['oom_kills']}, {usage['measured_with']}")
        finally:
            gone, _, _ = api.call("DELETE", f"/v1/volumes/{volume}")
        check("volume delete", gone == 200, f"delete after the sandbox -> {gone}")

    # ---- limits
    def endless_loop():
        sid = sandbox(512)
        started = time.time()
        status = run(api, sid, ["python", "-c", "while True: pass"], 5)
        took = time.time() - started
        api.json("DELETE", f"/v1/sandboxes/{sid}")
        left = api.json("GET", f"/v1/sandboxes?label=smoke:{tag['smoke']}")
        check("deadline", status["state"] == "timeout" and took <= 10 and not any(s["id"] == sid for s in left),
              f"state {status['state']} after {took:.1f} s (deadline 5 s), sandbox gone after delete")

    def out_of_memory():
        sid = sandbox(1024)
        status = run(api, sid, ["python", "-c", OOM_PROBE], 120)
        api.json("DELETE", f"/v1/sandboxes/{sid}")
        last = [line for line in status["stdout"].split("\n") if line][-1:]
        check("oom", status["oom_killed"], f"oom_killed {status['oom_killed']}, exit {status['exit_code']}, "
              f"state {status['state']} after {last}")

    def budget():
        capacity = api.json("GET", "/v1/capacity")
        size = api.json("GET", "/v1/info")["limits"]["max_sandbox_memory_mb"]
        created, refused, retry = 0, None, None
        try:
            for _ in range(10):
                body = {"image": "automl-runner", "cpus": 1, "memory_mb": size, "work_mb": 64, "labels": tag}
                status, headers, _ = api.call("POST", "/v1/sandboxes", body)
                if status != 201:
                    refused, retry = status, headers.get("Retry-After")
                    break
                created += 1
        finally:
            for s in api.json("GET", f"/v1/sandboxes?label=smoke:{tag['smoke']}"):
                api.call("DELETE", f"/v1/sandboxes/{s['id']}")
        fits = capacity["memory_free_mb"] // size
        check("budget", refused == 429 and retry is not None and created == fits,
              f"{created} x {size} MB fit in {capacity['memory_free_mb']} MB free, then {refused} (Retry-After {retry})")

    def leftovers():
        for s in api.json("GET", f"/v1/sandboxes?label=smoke:{tag['smoke']}"):
            api.call("DELETE", f"/v1/sandboxes/{s['id']}")
        counts = []
        for what in (["ps", "-aq"], ["volume", "ls", "-q"]):
            out = subprocess.run(["docker", *what, "--filter", "label=automl.sandbox.kind"],
                                 capture_output=True, text=True, check=True).stdout
            counts.append(len(out.split()))
        check("leftovers", counts == [0, 0], f"{counts[0]} containers, {counts[1]} volumes")

    for name, test in (("auth", auth), ("info", info), ("replay", replay_attempt), ("deadline", endless_loop),
                       ("oom", out_of_memory), ("budget", budget), ("leftovers", leftovers)):
        attempt(name, test)

    print("\n===== SMOKE SUMMARY (copy from here down to END) =====")
    print("\n".join(notes + results))
    print("===== END =====")


if __name__ == "__main__":
    main()
