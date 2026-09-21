"""Every Docker call: volumes, sandboxes, files, execs, admission control, usage, the reaper."""

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import docker
from docker.errors import APIError, DockerException
from docker.errors import NotFound as DockerNotFound

from automl_sandbox import __version__, cgroup
from automl_sandbox.cgroup import MB, Reading
from automl_sandbox.config import Config
from automl_sandbox.errors import Conflict, NoCapacity, NotFound
from automl_sandbox.output import OutputBuffer

log = logging.getLogger("automl_sandbox")

# the server only ever lists, changes or removes objects carrying KIND
KIND = "automl.sandbox.kind"
ID = "automl.sandbox.id"
CREATED = "automl.sandbox.created_at"
EXPIRES = "automl.sandbox.expires_at"
MEMORY = "automl.sandbox.memory_mb"
CPUS = "automl.sandbox.cpus"
USER_LABEL = "automl.label."
HASH_LABEL = "automl.runtime_hash"

HELPER_TTL = 600  # upload helpers and image probes
ORPHAN_AGE = 60  # a scratch volume this old without its sandbox is removed
PACKAGES = "import json, importlib.metadata as m; print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"


def new_id() -> str:
    return secrets.token_hex(6)


@dataclass
class Exec:
    id: str
    sandbox_id: str
    container_id: str
    stdout: OutputBuffer
    stderr: OutputBuffer
    oom_before: Optional[int]
    started: float = field(default_factory=time.monotonic)
    ended: Optional[float] = None
    state: str = "running"
    exit_code: Optional[int] = None
    oom_killed: bool = False
    timer: Optional[threading.Timer] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def finish(self, state: str, exit_code: Optional[int] = None) -> bool:
        with self.lock:
            if self.state != "running":
                return False
            self.state, self.exit_code, self.ended = state, exit_code, time.monotonic()
            return True


class Engine:
    def __init__(self, config: Config, client=None, stream_client=None):
        self.config = config
        self.client = client or docker.from_env(timeout=60)
        # exec output is read over a client with no read timeout: attempts run for hours
        self.stream_client = stream_client or docker.from_env(timeout=None)
        self.admission = threading.Lock()
        self.execs: dict[str, Exec] = {}
        self.readings: dict[str, tuple[Reading, str]] = {}  # latest usage per sandbox
        self.packages: dict[str, dict] = {}  # per image id

    # ---- host

    def info(self) -> dict:
        cfg = self.config
        cgroup_v2 = (Path(cfg.cgroup_root) / "cgroup.controllers").exists()
        return {
            "api_version": "1",
            "server_version": __version__,
            "runtime": cfg.runtime,
            "isolation": "gvisor" if cfg.runtime == "runsc" else "runc",
            "measurement": "cgroup-v2" if cgroup_v2 else "docker-stats",
            "images": {name: self._image_info(tag) for name, tag in cfg.images.items()},
            "limits": {
                "max_sandbox_memory_mb": cfg.max_sandbox_memory_mb,
                "max_cpus": cfg.max_cpus,
                "max_work_mb": cfg.max_work_mb,
                "max_exec_seconds": cfg.max_exec_seconds,
                "max_ttl_seconds": cfg.max_ttl_seconds,
            },
        }

    def _image_info(self, tag: str) -> dict:
        try:
            image = self.client.images.get(tag)
        except DockerNotFound:
            return {"tag": tag, "error": "image not found"}
        if image.id not in self.packages:
            out = self.client.containers.run(
                tag, ["python", "-c", PACKAGES], remove=True, network_mode="none",
                labels=self._labels("probe", new_id(), HELPER_TTL),
            )
            self.packages[image.id] = json.loads(out)
        return {"tag": tag, "runtime_hash": image.labels.get(HASH_LABEL), "packages": self.packages[image.id]}

    def capacity(self) -> dict:
        running = self._running()
        budget = self.config.memory_budget_mb
        reserved = sum(int(c.labels[MEMORY]) for c in running)
        return {
            "cpus": self.client.info()["NCPU"],
            "memory_budget_mb": budget,
            "memory_reserved_mb": reserved,
            "memory_free_mb": max(budget - reserved, 0),
            "sandboxes_running": len(running),
        }

    # ---- run volumes

    def create_volume(self, labels: dict, ttl: int) -> dict:
        vid = new_id()
        tags = self._labels("volume", vid, ttl, labels)
        self.client.volumes.create(name=f"automl-vol-{vid}", labels=tags)
        return {"id": vid, "expires_at": int(tags[EXPIRES])}

    def put_volume_files(self, vid: str, archive: bytes) -> None:
        volume = self._volume(vid)
        hid = new_id()
        # created, never started: only a way to reach the volume through the archive API
        helper = self.client.containers.create(
            next(iter(self.config.images.values())), ["true"], name=f"automl-helper-{hid}",
            labels=self._labels("helper", hid, HELPER_TTL), network_mode="none",
            volumes={volume.name: {"bind": "/data", "mode": "rw"}},
        )
        try:
            helper.put_archive("/data", archive)
        finally:
            helper.remove(force=True)

    def delete_volume(self, vid: str) -> None:
        self._volume(vid).remove()  # Docker answers 409 while a sandbox still mounts it

    # ---- sandboxes

    def create_sandbox(self, image: str, cpus: float, memory_mb: int, work_mb: int,
                       ttl: int, volume: Optional[str], labels: dict) -> dict:
        cfg = self.config
        data = self._volume(volume).name if volume else None
        sid = new_id()
        uid, gid = cfg.sandbox_user.split(":")
        tags = self._labels("sandbox", sid, ttl, labels)
        tags.update({MEMORY: str(memory_mb), CPUS: str(cpus)})

        with self.admission:
            reserved = sum(int(c.labels[MEMORY]) for c in self._running())
            if reserved + memory_mb > cfg.memory_budget_mb:
                raise NoCapacity(f"memory budget: {reserved} of {cfg.memory_budget_mb} MB reserved, {memory_mb} requested")

            work = self.client.volumes.create(
                name=f"automl-work-{sid}", driver="local",
                driver_opts={"type": "tmpfs", "device": "tmpfs", "o": f"size={work_mb}m,uid={uid},gid={gid},mode=0700"},
                labels=self._labels("work", sid, ttl),
            )
            mounts = {work.name: {"bind": "/work", "mode": "rw"}}
            if data:
                mounts[data] = {"bind": "/data", "mode": "ro"}
            try:
                self.client.containers.run(
                    cfg.images[image], ["sleep", "infinity"], name=f"automl-sbx-{sid}", detach=True,
                    labels=tags, runtime=cfg.runtime, network_mode="none", read_only=True,
                    tmpfs={"/tmp": f"rw,size={cfg.tmp_mb}m,mode=1777"}, volumes=mounts, working_dir="/work",
                    user=cfg.sandbox_user, cap_drop=["ALL"], security_opt=["no-new-privileges"],
                    pids_limit=cfg.pids_limit, mem_limit=f"{memory_mb}m", memswap_limit=f"{memory_mb}m",
                    # CPU is shared, not partitioned: the ceiling is the per-sandbox cap every
                    # request is already checked against, and the weight settles the split when
                    # they are all busy, so an idle neighbour's cores go to whoever can use them.
                    # Memory cannot work this way - past its limit the kernel kills the process.
                    nano_cpus=int(cfg.max_cpus * 1e9),
                    cpu_shares=max(2, int(1024 * cpus / cfg.max_cpus)),
                )
            except Exception:
                self._remove_sandbox(sid)
                raise
        return {"id": sid, "expires_at": int(tags[EXPIRES])}

    def list_sandboxes(self, labels: list) -> list:
        filters = [f"{KIND}=sandbox"] + [f"{USER_LABEL}{k}={v}" for k, v in labels]
        found = self.client.containers.list(all=True, filters={"label": filters}, ignore_removed=True)
        return [self._describe(c) for c in found]

    def put_files(self, sid: str, archive: bytes) -> None:
        self._running_sandbox(sid).put_archive("/work", archive)

    def get_files(self, sid: str, path: str):
        stream, _ = self._running_sandbox(sid).get_archive(f"/work/{path}")
        return stream

    def kill(self, sid: str) -> None:
        self._stop(self._sandbox(sid), "killed")

    def delete_sandbox(self, sid: str) -> dict:
        container = self._sandbox(sid)
        if container.status == "running":
            self._measure(sid, container.id)
        usage = self._usage(sid, container.labels)
        self._remove_sandbox(sid)
        return usage

    # ---- execs

    def start_exec(self, sid: str, cmd: list, deadline: float) -> str:
        container = self._running_sandbox(sid)
        before = self._measure(sid, container.id)
        api = self.stream_client.api
        docker_id = api.exec_create(container.id, cmd, user=self.config.sandbox_user, workdir="/work")["Id"]
        limit = self.config.max_output_kb * 1024
        ex = Exec(new_id(), sid, container.id, OutputBuffer(limit), OutputBuffer(limit),
                  before.oom_kills if before else None)
        ex.timer = threading.Timer(deadline, self._on_deadline, args=(ex,))
        ex.timer.daemon = True
        self.execs[ex.id] = ex
        ex.timer.start()
        threading.Thread(target=self._follow, args=(ex, docker_id), daemon=True).start()
        return ex.id

    def get_exec(self, sid: str, exec_id: str, stdout_offset: int, stderr_offset: int) -> dict:
        ex = self.execs.get(exec_id)
        if ex is None or ex.sandbox_id != sid:
            raise NotFound(f"no exec {exec_id} in sandbox {sid}")
        stdout, stdout_next, stdout_skipped = ex.stdout.read(stdout_offset)
        stderr, stderr_next, stderr_skipped = ex.stderr.read(stderr_offset)
        live = self._measure(sid, ex.container_id) if ex.state == "running" else None
        return {
            "state": ex.state,
            "exit_code": ex.exit_code,
            "oom_killed": ex.oom_killed,
            "elapsed_seconds": round((ex.ended or time.monotonic()) - ex.started, 1),
            "stdout": stdout, "stdout_offset": stdout_next, "stdout_skipped": stdout_skipped,
            "stderr": stderr, "stderr_offset": stderr_next, "stderr_skipped": stderr_skipped,
            "live": {"cpu_seconds": round(live.cpu_seconds, 2), "memory_mb": round(live.memory_mb, 1)} if live else None,
        }

    def _follow(self, ex: Exec, docker_id: str) -> None:
        api = self.stream_client.api
        exit_code = None
        try:
            for out, err in api.exec_start(docker_id, stream=True, demux=True):
                if out:
                    ex.stdout.write(out)
                if err:
                    ex.stderr.write(err)
            for _ in range(20):  # the exit code can trail the end of the stream
                info = api.exec_inspect(docker_id)
                if not info["Running"]:
                    exit_code = info["ExitCode"]
                    break
                time.sleep(0.1)
        except DockerException as error:
            log.warning("exec %s lost: %s", ex.id, error)
        after = self._measure(ex.sandbox_id, ex.container_id)
        ex.oom_killed = bool(after and ex.oom_before is not None and after.oom_kills > ex.oom_before)
        ex.finish("exited" if exit_code is not None else "lost", exit_code)
        ex.timer.cancel()

    def _on_deadline(self, ex: Exec) -> None:
        if ex.finish("timeout"):
            try:
                self._stop(self._sandbox(ex.sandbox_id), "killed")
            except (NotFound, DockerException) as error:
                log.warning("deadline kill for exec %s failed: %s", ex.id, error)

    def _stop(self, container, state: str) -> None:
        sid = container.labels[ID]
        self._measure(sid, container.id)  # the cgroup disappears with the container
        for ex in list(self.execs.values()):
            if ex.sandbox_id == sid:
                ex.finish(state)
        try:
            container.kill()
        except APIError as error:
            if error.status_code != 409:  # 409: already stopped
                raise

    # ---- usage

    def _measure(self, sid: str, container_id: str) -> Optional[Reading]:
        path = cgroup.find_dir(self.config.cgroup_root, container_id)
        reading, method = (cgroup.read(path) if path else None), "cgroup-v2"
        if reading is None:
            reading, method = self._docker_stats(container_id), "docker-stats"
        if reading is None:
            return None
        previous = self.readings.get(sid)
        if previous:
            reading.peak_memory_mb = max(reading.peak_memory_mb, previous[0].peak_memory_mb)
        self.readings[sid] = (reading, method)
        return reading

    def _docker_stats(self, container_id: str) -> Optional[Reading]:
        # for hosts without readable cgroup files, e.g. Docker Desktop
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False, one_shot=True)
            memory = stats["memory_stats"]
            current = memory["usage"] / MB
            return Reading(
                cpu_seconds=stats["cpu_stats"]["cpu_usage"]["total_usage"] / 1e9,
                memory_mb=current,
                peak_memory_mb=memory.get("max_usage", 0) / MB or current,
                oom_kills=int(container.attrs["State"]["OOMKilled"]),
            )
        except (DockerException, KeyError, TypeError):
            return None

    def _usage(self, sid: str, labels: dict) -> dict:
        reading, method = self.readings.get(sid, (None, None))
        wall = time.time() - float(labels[CREATED])
        memory_mb, cpus = int(labels[MEMORY]), float(labels[CPUS])
        return {
            "wall_seconds": round(wall, 1),
            "cpu_seconds": round(reading.cpu_seconds, 2) if reading else None,
            "peak_memory_mb": round(reading.peak_memory_mb, 1) if reading else None,
            "oom_kills": reading.oom_kills if reading else None,
            "memory_limit_mb": memory_mb,
            "cpus": cpus,
            "reserved_gb_seconds": round(memory_mb / 1024 * wall, 1),
            "reserved_cpu_seconds": round(cpus * wall, 1),
            "measured_with": method,
        }

    # ---- reaper

    def reap_forever(self) -> None:
        while True:
            time.sleep(self.config.reap_interval_seconds)
            try:
                self.reap()
            except Exception:
                log.exception("reaper failed")

    def reap(self) -> None:
        now = time.time()
        alive = set()
        for c in self.client.containers.list(all=True, filters={"label": KIND}, ignore_removed=True):
            labels = c.labels
            expired = int(labels[EXPIRES]) < now
            if labels[KIND] != "sandbox":
                if expired:
                    c.remove(force=True)
            elif expired:
                log.info("reaping expired sandbox %s", labels[ID])
                self._remove_sandbox(labels[ID])
            else:
                alive.add(labels[ID])

        for v in self.client.volumes.list(filters={"label": KIND}):
            labels = v.attrs.get("Labels") or {}
            if labels.get(KIND) == "work":
                stale = labels[ID] not in alive and float(labels[CREATED]) + ORPHAN_AGE < now
            else:
                stale = int(labels[EXPIRES]) < now
            if stale:
                try:
                    v.remove(force=True)
                except APIError as error:
                    log.info("volume %s kept: %s", v.name, error.explanation)

    # ---- helpers

    def _labels(self, kind: str, oid: str, ttl: int, user: Optional[dict] = None) -> dict:
        now = time.time()
        labels = {KIND: kind, ID: oid, CREATED: f"{now:.3f}", EXPIRES: str(int(now + ttl))}
        labels.update({USER_LABEL + k: v for k, v in (user or {}).items()})
        return labels

    def _describe(self, c) -> dict:
        labels = c.labels
        return {
            "id": labels[ID],
            "state": c.status,
            "labels": {k[len(USER_LABEL):]: v for k, v in labels.items() if k.startswith(USER_LABEL)},
            "memory_mb": int(labels[MEMORY]),
            "cpus": float(labels[CPUS]),
            "created_at": int(float(labels[CREATED])),
            "expires_at": int(labels[EXPIRES]),
        }

    def _running(self) -> list:
        return self.client.containers.list(filters={"label": f"{KIND}=sandbox", "status": "running"}, ignore_removed=True)

    def _sandbox(self, sid: str):
        try:
            container = self.client.containers.get(f"automl-sbx-{sid}")
        except DockerNotFound:
            raise NotFound(f"no sandbox {sid}")
        if container.labels.get(KIND) != "sandbox":
            raise NotFound(f"no sandbox {sid}")
        return container

    def _running_sandbox(self, sid: str):
        container = self._sandbox(sid)
        if container.status != "running":
            raise Conflict(f"sandbox {sid} is {container.status}, not running")
        return container

    def _volume(self, vid: str):
        try:
            volume = self.client.volumes.get(f"automl-vol-{vid}")
        except DockerNotFound:
            raise NotFound(f"no volume {vid}")
        if (volume.attrs.get("Labels") or {}).get(KIND) != "volume":
            raise NotFound(f"no volume {vid}")
        return volume

    def _remove_sandbox(self, sid: str) -> None:
        try:
            self.client.containers.get(f"automl-sbx-{sid}").remove(force=True)
        except DockerNotFound:
            pass
        try:
            self.client.volumes.get(f"automl-work-{sid}").remove(force=True)
        except DockerNotFound:
            pass
        for exec_id in [e.id for e in self.execs.values() if e.sandbox_id == sid]:
            self.execs.pop(exec_id).timer.cancel()
        self.readings.pop(sid, None)
