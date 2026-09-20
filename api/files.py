"""
files.py
--------
Where an uploaded CSV lives, and the 100 MB cap (D3).

For the beta the API and the worker share a machine, so an upload is written
straight to `uploads/{workspace}/{source}.csv` beside `runs/` and the worker
reads that path — no object store in between (docs/PHASE4.md §8.1). The cap is
enforced while the file streams in, so a file too big is refused partway
through rather than after it has all arrived.
"""

from __future__ import annotations

import os
from pathlib import Path

MAX_BYTES = 100 * 1024 * 1024
UPLOADS_ROOT = Path(os.environ.get("AUTOML_UPLOADS_ROOT") or "uploads")


class TooLarge(Exception):
    """The upload went past MAX_BYTES; whatever was written has been removed."""


def path_for(workspace_id: int, source_id: int) -> Path:
    return UPLOADS_ROOT / str(workspace_id) / f"{source_id}.csv"


async def write(path: Path, chunks) -> int:
    """Stream chunks to path; returns the size. Past the cap: delete and raise TooLarge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with path.open("wb") as handle:
            async for chunk in chunks:
                written += len(chunk)
                if written > MAX_BYTES:
                    raise TooLarge
                handle.write(chunk)
    except BaseException:
        remove(path)
        raise
    return written


def ours(path: Path | str | None) -> bool:
    """Whether a path is one of our uploads, and so ours to delete.

    sources.path can hold a file we didn't write — `cli enqueue` points a source
    at a try-out CSV in the repo — and paths come back out of the database, so
    every delete asks this first (worker/sweep.py asks the same question).
    """
    if not path:
        return False
    root = UPLOADS_ROOT.resolve()
    try:
        full = Path(path).resolve()
    except OSError:
        return False
    return full != root and full.is_relative_to(root)


def remove(path: Path | str | None) -> None:
    """Delete an upload. A path outside the uploads folder is left alone."""
    if ours(path):
        Path(path).unlink(missing_ok=True)
