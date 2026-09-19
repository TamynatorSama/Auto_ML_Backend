"""Reads and checks config.toml."""

import tomllib
from dataclasses import dataclass, field, fields

RUNTIMES = ("runsc", "runc")


@dataclass(frozen=True)
class Config:
    token: str
    runtime: str = "runsc"
    memory_budget_mb: int = 8192
    max_sandbox_memory_mb: int = 6144
    max_cpus: float = 4.0
    pids_limit: int = 512
    max_work_mb: int = 4096
    tmp_mb: int = 512
    default_ttl_seconds: int = 3600
    max_ttl_seconds: int = 86400
    max_exec_seconds: int = 7200
    max_upload_mb: int = 200
    max_output_kb: int = 2048
    sandbox_user: str = "1000:1000"
    cgroup_root: str = "/sys/fs/cgroup"
    reap_interval_seconds: int = 30
    images: dict = field(default_factory=lambda: {"automl-runner": "automl-runner:current"})


def load(path) -> Config:
    with open(path, "rb") as handle:
        return from_dict(tomllib.load(handle))


def from_dict(data: dict) -> Config:
    known = {f.name: f for f in fields(Config)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(unknown)}")
    if "token" not in data:
        raise ValueError("token is required")

    for name, value in data.items():
        want = known[name].type
        if type(value) is not want and not (want is float and type(value) is int):
            raise ValueError(f"{name} must be {want.__name__}")

    config = Config(**data)
    if len(config.token) < 32:
        raise ValueError("token must be at least 32 characters")
    if config.runtime not in RUNTIMES:
        raise ValueError(f"runtime must be one of {', '.join(RUNTIMES)}")
    uid, _, gid = config.sandbox_user.partition(":")
    if not (uid.isdigit() and gid.isdigit()) or int(uid) == 0:
        raise ValueError("sandbox_user must be numeric uid:gid, and not root")
    if not config.images or not all(isinstance(v, str) for v in config.images.values()):
        raise ValueError("images must map names to image tags")
    return config
