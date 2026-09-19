import io
import tarfile

import pytest

from automl_sandbox.errors import BadRequest
from automl_sandbox.tarsafe import clean
from conftest import make_tar


def member(name, kind=tarfile.REGTYPE, **attrs):
    info = tarfile.TarInfo(name)
    info.type = kind
    for key, value in attrs.items():
        setattr(info, key, value)
    return info


def test_counts_files_and_rewrites_owner():
    archive = make_tar([member("models", tarfile.DIRTYPE, mode=0o755), ("models/m.txt", b"abc"), ("./c.py", b"xy")])
    cleaned, files, size = clean(archive, "1000:1000")
    assert (files, size) == (2, 5)
    with tarfile.open(fileobj=io.BytesIO(cleaned)) as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == ["models", "models/m.txt", "c.py"]
        assert all((m.uid, m.gid, m.uname, m.gname) == (1000, 1000, "", "") for m in members)
        assert tar.extractfile("models/m.txt").read() == b"abc"


def test_strips_setuid_and_pax_owner_override():
    info = member("run.sh", mode=0o4755, uid=0, pax_headers={"uid": "0", "uname": "root"})
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(info, io.BytesIO(b""))
    cleaned, _, _ = clean(out.getvalue(), "1000:1000")
    with tarfile.open(fileobj=io.BytesIO(cleaned)) as tar:
        (m,) = tar.getmembers()
    assert (m.mode, m.uid, m.uname) == (0o755, 1000, "")


@pytest.mark.parametrize("entry", [
    ("/etc/passwd", b"x"),
    ("../escape.py", b"x"),
    ("a/../../escape.py", b"x"),
    member(".", tarfile.DIRTYPE),
    member("./", tarfile.DIRTYPE),
    member("link", tarfile.SYMTYPE, linkname="/etc"),
    member("hard", tarfile.LNKTYPE, linkname="c.py"),
    member("dev", tarfile.CHRTYPE),
    member("fifo", tarfile.FIFOTYPE),
])
def test_refuses_unsafe_entries(entry):
    with pytest.raises(BadRequest):
        clean(make_tar([entry]), "1000:1000")


def test_refuses_non_tar():
    with pytest.raises(BadRequest):
        clean(b"not a tar archive at all" * 40, "1000:1000")
