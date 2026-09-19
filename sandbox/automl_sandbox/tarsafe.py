"""Checks an uploaded tar archive and rewrites it so it can't escape its destination."""

import io
import tarfile
from pathlib import PurePosixPath

from automl_sandbox.errors import BadRequest


def clean(data: bytes, user: str):
    """Files and folders only, owned by user -> (archive, files, bytes)."""
    uid, gid = (int(part) for part in user.split(":"))
    out = io.BytesIO()
    files = size = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as src, tarfile.open(fileobj=out, mode="w") as dst:
            for member in src:
                path = PurePosixPath(member.name)
                if member.name.startswith("/") or ".." in path.parts:
                    raise BadRequest(f"unsafe path in archive: {member.name}")
                if not path.parts:
                    # extracting "." as root would reset the owner of the destination
                    raise BadRequest("archive may not contain an entry for '.'")
                if not (member.isfile() or member.isdir()):
                    raise BadRequest(f"only files and folders are allowed: {member.name}")
                if member.issparse():
                    # a few bytes of sparse map can declare gigabytes of zeros
                    raise BadRequest(f"sparse files are not allowed: {member.name}")

                member.name = str(path)
                member.uid, member.gid, member.uname, member.gname = uid, gid, "", ""
                member.mode &= 0o777
                member.pax_headers = {}  # they would override the fields above
                if member.isfile():
                    files += 1
                    size += member.size
                    dst.addfile(member, src.extractfile(member))
                else:
                    dst.addfile(member)
    except tarfile.TarError as error:
        raise BadRequest(f"not a valid uncompressed tar archive: {error}")
    return out.getvalue(), files, size
