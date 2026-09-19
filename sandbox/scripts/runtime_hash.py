"""Content hash of automl_runtime: the version handshake between worker and sandbox host.

    python3 sandbox/scripts/runtime_hash.py [path/to/automl_runtime]
"""

import hashlib
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "automl_runtime"


def runtime_hash(root: Path = DEFAULT) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(root).glob("*.py")):
        digest.update(path.name.encode() + b"\0")
        # CRLF -> LF so a Windows checkout hashes the same as Linux
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


if __name__ == "__main__":
    print(runtime_hash(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
