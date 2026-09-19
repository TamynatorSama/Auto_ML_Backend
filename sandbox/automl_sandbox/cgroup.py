"""CPU time, memory and OOM kills from a container's cgroup v2 files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MB = 1024 * 1024


@dataclass
class Reading:
    cpu_seconds: float
    memory_mb: float
    peak_memory_mb: float
    oom_kills: int


def find_dir(root: str, container_id: str) -> Optional[Path]:
    # systemd cgroup driver first, then cgroupfs
    for rel in (f"system.slice/docker-{container_id}.scope", f"docker/{container_id}"):
        path = Path(root) / rel
        if path.is_dir():
            return path
    return None


def _keyed(path: Path) -> dict:
    pairs = (line.split() for line in path.read_text().splitlines())
    return {p[0]: int(p[1]) for p in pairs if len(p) == 2}


def read(path: Path) -> Optional[Reading]:
    try:
        current = int((path / "memory.current").read_text()) / MB
        peak_file = path / "memory.peak"
        peak = int(peak_file.read_text()) / MB if peak_file.exists() else current
        return Reading(
            cpu_seconds=_keyed(path / "cpu.stat")["usage_usec"] / 1e6,
            memory_mb=current,
            peak_memory_mb=peak,
            oom_kills=_keyed(path / "memory.events").get("oom_kill", 0),
        )
    except (OSError, ValueError, KeyError):
        return None
