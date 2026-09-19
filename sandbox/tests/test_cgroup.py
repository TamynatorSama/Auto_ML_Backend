from automl_sandbox import cgroup

MB = 1024 * 1024


def fake_cgroup(path, peak=True):
    path.mkdir(parents=True)
    (path / "cpu.stat").write_text("usage_usec 12110000\nuser_usec 10000000\nsystem_usec 2110000\n")
    (path / "memory.current").write_text(f"{100 * MB}\n")
    if peak:
        (path / "memory.peak").write_text(f"{193 * MB}\n")
    (path / "memory.events").write_text("low 0\nhigh 0\nmax 12\noom 1\noom_kill 1\noom_group_kill 0\n")
    return path


def test_finds_systemd_then_cgroupfs(tmp_path):
    assert cgroup.find_dir(str(tmp_path), "abc") is None
    cgroupfs = fake_cgroup(tmp_path / "docker" / "abc")
    assert cgroup.find_dir(str(tmp_path), "abc") == cgroupfs
    systemd = fake_cgroup(tmp_path / "system.slice" / "docker-abc.scope")
    assert cgroup.find_dir(str(tmp_path), "abc") == systemd


def test_reads_usage(tmp_path):
    reading = cgroup.read(fake_cgroup(tmp_path / "c"))
    assert reading == cgroup.Reading(cpu_seconds=12.11, memory_mb=100.0, peak_memory_mb=193.0, oom_kills=1)


def test_without_memory_peak_uses_current(tmp_path):
    assert cgroup.read(fake_cgroup(tmp_path / "c", peak=False)).peak_memory_mb == 100.0


def test_missing_files_give_none(tmp_path):
    path = fake_cgroup(tmp_path / "c")
    (path / "cpu.stat").unlink()
    assert cgroup.read(path) is None
    assert cgroup.read(tmp_path / "nowhere") is None
