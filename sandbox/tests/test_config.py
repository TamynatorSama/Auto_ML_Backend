import pytest

from automl_sandbox.config import from_dict, load

TOKEN = "t" * 32


def test_defaults():
    config = from_dict({"token": TOKEN})
    assert (config.runtime, config.memory_budget_mb, config.pids_limit) == ("runsc", 8192, 512)
    assert config.images == {"automl-runner": "automl-runner:current"}


def test_loads_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(f'token = "{TOKEN}"\nruntime = "runc"\nmax_cpus = 2\n[images]\nr = "r:1"\n')
    config = load(path)
    assert (config.runtime, config.max_cpus, config.images) == ("runc", 2, {"r": "r:1"})


@pytest.mark.parametrize("data, message", [
    ({}, "token is required"),
    ({"token": "short"}, "at least 32"),
    ({"token": TOKEN, "tokn": 1}, "unknown settings: tokn"),
    ({"token": TOKEN, "runtime": "kata"}, "runtime must be"),
    ({"token": TOKEN, "memory_budget_mb": "8G"}, "memory_budget_mb must be int"),
    ({"token": TOKEN, "pids_limit": True}, "pids_limit must be int"),
    ({"token": TOKEN, "images": {}}, "images"),
    ({"token": TOKEN, "images": {"r": 3}}, "images"),
])
def test_refuses_bad_config(data, message):
    with pytest.raises(ValueError, match=message):
        from_dict(data)
