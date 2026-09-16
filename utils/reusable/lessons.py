"""
lessons.py
----------
What this run has already learned about getting candidates to run.

Several models in one run make the same mechanical mistakes, because they are
written by the same generator against the same library versions: an import
from the wrong module, a transformer signature, an encoder argument. A fix one
worker found protects the others only if it is written down where they read.
Each worker's fixes used to live in its own prompt history, and even there only
within one generation, so the same error was repaired again and again.

A lesson is recorded when a repair gets an attempt past the stage that failed:
the normalised error it fixed, and the change that fixed it. Every generator,
judge and fixer prompt in the run then carries the list.

    error_signature(record)                 -> str | None
    got_past(failed, repaired)              -> bool
    record_lesson(run_dir, model, failed, repaired) -> dict | None
    read_lessons(run_dir) / render_lessons(lessons)
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import List, Optional

from models import AttemptRecord

LESSONS_FILE = "lessons.json"
PROMPT_LESSONS = 12          # most recent distinct lessons shown in a prompt

# the evaluator's ladder, in order; a later stage means an earlier one passed
STAGES = (
    "preflight", "import", "data", "build", "estimator", "smoke", "pickle",
    "cv", "budget", "scoring", "importance",
)

_STAGE = re.compile(r"\[stage: ([a-z_]+)\]")
_EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))\b:?\s*(.*)$")
_PATH = re.compile(r"\s*\((?:[A-Za-z]:)?[\\/][^)]*\)")
_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")
_NUMBER = re.compile(r"(?<![\w'])\d+(?:\.\d+)?(?![\w'])")

_lock = threading.Lock()


def stage_of(record: AttemptRecord) -> Optional[str]:
    match = _STAGE.search(record.traceback or "")
    return match.group(1) if match else None


def error_signature(record: AttemptRecord) -> Optional[str]:
    """The failure with everything incidental removed: stage, exception, message.

    File paths, memory addresses and numbers vary between attempts that fail for
    the same reason; names in quotes do not, and they are what the fix is about.
    """
    if record.status not in ("error", "syntax_error", "dependency_error", "timeout", "out_of_memory"):
        return None
    text = (record.traceback or "").strip()
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip() and not _STAGE.fullmatch(line.strip())]
    message = next((line for line in reversed(lines) if _EXCEPTION.match(line)), lines[-1] if lines else "")
    message = _PATH.sub("", message)
    message = _ADDRESS.sub("", message)
    message = _NUMBER.sub("N", message)
    return f"{stage_of(record) or record.status}: {message}"[:240]


def _stage_index(record: AttemptRecord) -> int:
    stage = stage_of(record)
    return STAGES.index(stage) if stage in STAGES else -1


def got_past(failed: AttemptRecord, repaired: AttemptRecord) -> bool:
    """Did the repair clear the stage the failure stopped at?"""
    if repaired.status in ("ok", "cached"):
        return True
    if repaired.status not in ("error", "syntax_error"):
        return False
    before, after = _stage_index(failed), _stage_index(repaired)
    return before >= 0 and after > before


def read_lessons(run_dir: str | Path) -> List[dict]:
    path = Path(run_dir) / LESSONS_FILE
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []


def record_lesson(
    run_dir: str | Path, model: str, failed: AttemptRecord, repaired: AttemptRecord
) -> Optional[dict]:
    """Write down a fix that worked, once per distinct error."""
    signature = error_signature(failed)
    if signature is None or not got_past(failed, repaired):
        return None

    fix = (repaired.changes or "").removeprefix("repair:").strip()
    lesson = {"signature": signature, "stage": stage_of(failed) or failed.status, "fix": fix[:300],
              "model": model, "attempt": repaired.attempt}
    path = Path(run_dir) / LESSONS_FILE
    with _lock:
        lessons = read_lessons(run_dir)
        if any(existing["signature"] == signature for existing in lessons):
            return None
        lessons.append(lesson)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(lessons, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    return lesson


def render_lessons(lessons: List[dict]) -> str:
    if not lessons:
        return ""
    lines = ["## KNOWN PITFALLS IN THIS RUN",
             "Errors other attempts in this run already hit, and the change that fixed each. "
             "Do not reintroduce them."]
    for lesson in lessons[-PROMPT_LESSONS:]:
        lines.append(f"- {lesson['signature']}\n  fixed by: {lesson['fix'] or 'unrecorded'}")
    return "\n".join(lines)
