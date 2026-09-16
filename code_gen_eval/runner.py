"""
runner.py
---------
Execute one generated training script and turn what happened into an
AttemptRecord.

The script never fails the run. A traceback, a timeout, or unparseable output
are all results: they are what the judge reads to write the next attempt, so
they come back as records rather than exceptions.

    run_script(code, model, attempt, context) -> AttemptRecord

Backends:
    subprocess  - runs in the current interpreter's environment. Fast, no
                  isolation. For developing against, not for real runs.
    docker      - one container per attempt, no network, capped cpu and memory.

The script is told where everything is through the environment, so it never
has to guess a path or care what directory it was started in:

    AUTOML_TRAIN, AUTOML_TEST, AUTOML_TARGET, AUTOML_TASK_TYPE, AUTOML_METRICS,
    AUTOML_CV_STRATEGY, AUTOML_CV_FOLDS, AUTOML_GROUP_COLUMN, AUTOML_DROP_COLUMNS,
    AUTOML_SEED, AUTOML_N_JOBS, AUTOML_MODEL, AUTOML_OUT, AUTOML_FINAL

The cross-validation settings are handed over rather than left to the script:
every model has to fold the training set the same way or their scores are not
comparable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys as _sys
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import psutil

from models import AttemptRecord, RunContext
from utils.reusable.guards import check_attempt

NEWLINE = chr(10)
RESULT_SENTINEL = "===AUTOML_RESULT==="
DOCKER_IMAGE = "automl-runner"
STDOUT_KEPT = 4000          # characters of stdout carried into the record
POLL_SECONDS = 0.1
ERROR_GRACE_SECONDS = 10     # how long a process may keep running after a traceback appears


def attempt_dir(context: RunContext, model: str, attempt: int) -> Path:
    """runs/{run_id}/{model}/attempt_{n} — one directory per attempt.

    Namespaced by model because workers run in parallel and would otherwise
    overwrite each other's scripts and output.
    """
    return Path(context.run_dir) / model / f"attempt_{attempt}"


def _script_environment(context: RunContext, model: str, out_dir: Path, final: bool) -> dict:
    environment = dict(os.environ)
    environment.update(
        {
            "AUTOML_TRAIN": str(Path(context.train_path).resolve()),
            "AUTOML_TEST": str(Path(context.test_path).resolve()),
            "AUTOML_TARGET": context.target,
            "AUTOML_SEED": str(context.split_plan.random_seed),
            "AUTOML_N_JOBS": str(context.n_jobs),
            "AUTOML_MODEL": model,
            "AUTOML_TASK_TYPE": context.task_type,
            "AUTOML_METRICS": ",".join(context.eval_matrics),
            "AUTOML_CV_STRATEGY": context.split_plan.cv_strategy,
            "AUTOML_CV_FOLDS": str(context.split_plan.cv_folds),
            "AUTOML_GROUP_COLUMN": context.split_plan.group_column or "",
            "AUTOML_DROP_COLUMNS": ",".join(context.drop_columns),
            "AUTOML_OUT": str(out_dir.resolve()),
            "AUTOML_FINAL": "1" if final else "0",
        }
    )
    return environment


def _parse_result(stdout: str) -> Optional[dict]:
    """Read the JSON block the script prints after the sentinel.

    A sentinel rather than "the last JSON-looking thing": training libraries
    print plenty of braces, and a silent mis-parse would put a wrong score on
    the leaderboard.
    """
    if RESULT_SENTINEL not in stdout:
        return None

    tail = stdout.rsplit(RESULT_SENTINEL, 1)[1]
    try:
        payload, _ = json.JSONDecoder().raw_decode(tail[tail.find("{"):])
        return payload
    except (ValueError, json.JSONDecodeError):
        return None


def _drain(stream, sink: list) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
    stream.close()


def _execute(command: list, cwd: Path, environment: dict, timeout: int):
    """Run a command, watching peak memory, and stop waiting once it is doomed.

    Output is drained by reader threads rather than at the end, so a traceback
    can be seen while the process is still alive. That matters because an error
    raised inside joblib's worker processes can leave the parent sitting there:
    an InvalidParameterError that takes milliseconds to hit was costing the
    entire time budget and being reported as a slow model.

    Returns (exit_code, stdout, stderr, wall_seconds, peak_mb, timed_out, died_after_error).
    """
    started = time.perf_counter()
    process = psutil.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    out_lines, err_lines = [], []
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, out_lines), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, err_lines), daemon=True),
    ]
    for reader in readers:
        reader.start()

    peak_bytes = 0
    timed_out = False
    died_after_error = False
    first_error_at = None

    while process.poll() is None:
        try:
            peak_bytes = max(peak_bytes, process.memory_info().rss)
            for child in process.children(recursive=True):
                peak_bytes = max(peak_bytes, child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        now = time.perf_counter()

        # a traceback has appeared but the process is still running. Give it a
        # moment to exit on its own — a library may recover — then stop waiting.
        if first_error_at is None and _has_traceback("".join(err_lines)):
            first_error_at = now
        elif first_error_at is not None and now - first_error_at > ERROR_GRACE_SECONDS:
            died_after_error = True

        if died_after_error or now - started > timeout:
            timed_out = not died_after_error
            for child in process.children(recursive=True):
                child.kill()
            process.kill()
            break

        time.sleep(POLL_SECONDS)

    process.wait()
    for reader in readers:
        reader.join(timeout=5)

    return (
        process.returncode,
        "".join(out_lines),
        "".join(err_lines),
        time.perf_counter() - started,
        peak_bytes / 1e6,
        timed_out,
        died_after_error,
    )


def _subprocess_command(out_dir: Path) -> list:
    return [sys.executable, "script.py"]


def _docker_command(context: RunContext, out_dir: Path, environment: dict) -> list:
    """One container per attempt: no network, capped cpu, memory and pids.

    The split is mounted read-only and only the attempt directory is writable,
    so a generated script cannot damage the dataset every other model shares.
    """
    data_dir = Path(context.train_path).resolve().parent
    container_environment = {
        key: value for key, value in environment.items() if key.startswith("AUTOML_")
    }
    container_environment["AUTOML_TRAIN"] = f"/data/{Path(context.train_path).name}"
    container_environment["AUTOML_TEST"] = f"/data/{Path(context.test_path).name}"
    container_environment["AUTOML_OUT"] = "/work"

    command = [
        "docker", "run", "--rm",
        "--network", "none",
        "--cpus", str(context.n_jobs),
        "--memory", "4g",
        "--pids-limit", "256",
        "-v", f"{data_dir}:/data:ro",
        "-v", f"{out_dir.resolve()}:/work",
        "-w", "/work",
    ]
    for key, value in container_environment.items():
        command += ["-e", f"{key}={value}"]
    command += [DOCKER_IMAGE, "python", "script.py"]
    return command


def _imported_modules(code: str) -> set:
    """Top-level module names the script imports, by parsing rather than running."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def check_syntax(code: str) -> Optional[str]:
    """Parse the script without running it.

    A SyntaxError costs a process spawn to discover and tells the next attempt
    nothing it could not have been told instantly. Reporting the line and the
    offending text is also a better repair prompt than the interpreter's own
    output buried in a traceback.
    """
    try:
        ast.parse(code)
    except SyntaxError as error:
        where = f"line {error.lineno}" + (f", column {error.offset}" if error.offset else "")
        offending = (error.text or "").rstrip()
        detail = f"{NEWLINE}    {offending}" if offending else ""
        return f"the script does not parse: {error.msg} ({where}){detail}"
    return None


def preflight(code: str, context: RunContext) -> Optional[str]:
    """Reject a script that cannot run, before spending a process on it.

    Two checks, both instant: does it parse, and does it import anything this
    environment does not have. Only third-party imports are checked — the
    standard library is whatever the interpreter ships with.
    """
    broken = check_syntax(code)
    if broken is not None:
        return broken

    if not context.environment:
        return None

    known = set(context.environment) | set(_sys.stdlib_module_names) | {"__future__"}
    unknown = sorted(module for module in _imported_modules(code) if module not in known)
    if not unknown:
        return None

    return (
        f"imports not available in this environment: {', '.join(unknown)}. "
        f"Available: {', '.join(sorted(context.environment))}."
    )


def _has_traceback(text: str) -> bool:
    return "Traceback (most recent call last)" in text or bool(
        re.search(r"^[A-Za-z_.]*(Error|Exception):", text, re.MULTILINE)
    )


_MISSING_MODULE = re.compile(r"No module named '([^']+)'")
_BAD_NAME = re.compile(r"cannot import name '[^']+' from '([^']+)'")


def _is_dependency_error(text: str, context: RunContext) -> bool:
    """A package that is not installed, as opposed to a wrong import path.

    `cannot import name 'TransformedTargetRegressor' from 'sklearn.preprocessing'`
    names a package that IS installed — the script simply looked in the wrong
    module, which the next attempt can fix. Calling that a dependency failure
    would abandon a model over a typo.
    """
    installed = set(context.environment)

    match = _MISSING_MODULE.search(text)
    if match:
        return match.group(1).split(".")[0] not in installed

    match = _BAD_NAME.search(text)
    if match:
        return match.group(1).split(".")[0] not in installed

    return False


def _cached_record(
    out_dir: Path, digest: str, model: str, attempt: int, prior_attempts: List[int]
) -> Optional[AttemptRecord]:
    """Reuse the result of an identical script run earlier in THIS loop.

    A retry that changes nothing is common — the judge asks for something the
    generator already did — and refitting it teaches nobody anything.

    Only the attempts the caller says it has already run are considered. Reading
    whatever happens to be on disk would match this attempt's own directory, and
    would reuse records left by an entirely different run.
    """
    for number in sorted(prior_attempts):
        sibling = out_dir.parent / f"attempt_{number}"
        if sibling == out_dir:
            continue
        hash_file = sibling / "script.sha256"
        record_file = sibling / "record.json"
        if not hash_file.exists() or not record_file.exists():
            continue
        if hash_file.read_text(encoding="utf-8").strip() != digest:
            continue

        record = AttemptRecord.model_validate_json(record_file.read_text(encoding="utf-8"))
        return record.model_copy(
            update={
                "attempt": attempt,
                "generation": generation,
                "kind": kind,
                "status": "cached",
                "script_path": str(out_dir / "script.py"),
                "changes": f"identical to attempt {record.attempt}; not re-run",
            }
        )
    return None


def run_script(
    code: str,
    model: str,
    attempt: int,
    context: RunContext,
    backend: str = "subprocess",
    final: bool = False,
    changes: str = "",
    prior_attempts: Optional[List[int]] = None,
    generation: int = 1,
    kind: str = "generate",
) -> AttemptRecord:
    out_dir = attempt_dir(context, model, attempt)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / "script.py"
    script_path.write_text(code, encoding="utf-8")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    (out_dir / "script.sha256").write_text(digest, encoding="utf-8")

    if not final and prior_attempts:
        cached = _cached_record(out_dir, digest, model, attempt, prior_attempts)
        if cached is not None:
            (out_dir / "record.json").write_text(cached.model_dump_json(indent=2), encoding="utf-8")
            return cached

    blocked = preflight(code, context)
    if blocked is not None:
        record = AttemptRecord(
            model=model,
            attempt=attempt,
            generation=generation,
            kind=kind,
            script_path=str(script_path),
            status="syntax_error" if blocked.startswith("the script does not parse") else "dependency_error",
            changes=changes,
            traceback=blocked,
        )
        (out_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        print(f"  preflight: {blocked}")
        return record

    environment = _script_environment(context, model, out_dir, final)
    if backend == "docker":
        command = _docker_command(context, out_dir, environment)
    elif backend == "subprocess":
        command = _subprocess_command(out_dir)
    else:
        raise ValueError(f"unknown backend: {backend}")

    exit_code, stdout, stderr, wall_seconds, peak_mb, timed_out, died_after_error = _execute(
        command, out_dir, environment, context.time_budget_seconds
    )

    (out_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    (out_dir / "command.txt").write_text(shlex.join(command), encoding="utf-8")

    record = AttemptRecord(
        model=model,
        attempt=attempt,
        generation=generation,
        kind=kind,
        script_path=str(script_path),
        changes=changes,
        stdout=stdout[-STDOUT_KEPT:],
        wall_seconds=round(wall_seconds, 3),
        peak_memory_mb=round(peak_mb, 1),
    )

    if died_after_error:
        # it raised and then sat there. The error is the result; waiting out the
        # rest of the budget would only have produced the same error later, and
        # calling it a timeout sends the judge after a cost problem that is not
        # there.
        record.status = "dependency_error" if _is_dependency_error(stderr, context) else "error"
        record.traceback = (
            f"raised after {record.wall_seconds}s and did not exit, so it was stopped:\n"
            + stderr[-STDOUT_KEPT:]
        )
        (out_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    if timed_out:
        # a real signal for the next attempt, not a crash: the model is too
        # expensive as configured and the judge can say so.
        #
        # Unless it is not. A process that raised before the budget expired was
        # broken, not slow — unpicklable pipeline steps hang joblib workers
        # exactly like an expensive fit does. Throwing stderr away here sends
        # the judge after a performance problem that does not exist.
        record.status = "timeout"
        note = f"killed after {context.time_budget_seconds}s (time budget)"
        if _has_traceback(stderr):
            note += (
                "\nthe process also reported errors before the budget expired, so this may be "
                "a failure rather than an expensive fit:\n" + stderr[-STDOUT_KEPT:]
            )
        elif stderr.strip():
            note += "\nstderr tail:\n" + stderr[-1000:]
        record.traceback = note
        (out_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    payload = _parse_result(stdout)

    if exit_code != 0:
        # a missing package is not a modelling failure and no amount of
        # rewriting the model will fix it, so it is kept as its own status
        record.status = "dependency_error" if _is_dependency_error(stderr, context) else "error"
        record.traceback = stderr[-STDOUT_KEPT:]
    elif payload is None:
        record.status = "error"
        record.traceback = (
            f"script exited 0 but printed no {RESULT_SENTINEL} block, so there is nothing to score"
        )
    else:
        record.cv_scores = {k: float(v) for k, v in (payload.get("cv_scores") or {}).items()}
        record.test_scores = {k: float(v) for k, v in (payload.get("test_scores") or {}).items()}
        record.fit_seconds = payload.get("fit_seconds")
        record.artifacts = {k: str(v) for k, v in (payload.get("artifacts") or {}).items()}
        # a clean exit and a well-formed result is not the same as a correct one
        record.warnings = check_attempt(record, context)
        for warning in record.warnings:
            print(f"  guard: {warning}")

    (out_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record
