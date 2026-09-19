"""
runner.py
---------
Evaluate one generated candidate and turn what happened into an AttemptRecord.

The candidate never fails the run. A traceback, a timeout, or an estimator that
cannot be scored are all results: they are what the judge and the fixer read,
so they come back as records rather than exceptions.

    run_candidate(code, model, attempt, context, final=False) -> AttemptRecord

The candidate only builds an estimator. Everything that decides what a score
means (loading the data, dropping excluded columns, the frozen folds, fitting,
pooled out-of-fold scoring, the test set, saving the model) is done by
`python -m automl_runtime.evaluate`, identically for every model. The runner
writes the candidate and its contract into the attempt directory, runs the
evaluator there, and reads back result.json.

Backends:
    subprocess  - the evaluator runs in this interpreter's environment. Fast, no
                  isolation.
    sandbox     - a fresh sandbox on the run's sandbox host for every attempt
                  (sandbox/DESIGN.md): no network, gVisor, capped cpu and memory.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import pkgutil
import re
import shlex
import subprocess
import sys
import sys as _sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

from automl_runtime.contract import Contract
from automl_runtime.evaluate import (
    CONTRACT_FILE,
    FINAL_RESULT_FILE,
    LOAD_CHECK_FILE,
    LOAD_CHECK_TIMEOUT,
    MODEL_FILE,
    RESULT_FILE,
)
from code_gen_eval.sandbox_client import SandboxError, SandboxUnavailable
from models import AttemptRecord, RunContext
from utils.reusable import hooks
from utils.reusable.guards import check_attempt
from utils.reusable.resources import memory_limit, wait_for_memory

NEWLINE = chr(10)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_FILE = "candidate.py"
BUILD_FUNCTION = "build_pipeline"
RUNNER_IMAGE = "automl-runner"   # a name from the sandbox host's image allowlist
WORK_MB = 2048                  # /work in a sandbox: the attempt's files and outputs
VOLUME_TTL = 86400              # the run's data stays on the host at most a day
SANDBOX_POLL_SECONDS = 0.5
STDOUT_KEPT = 4000          # characters of stdout carried into the record
POLL_SECONDS = 0.1
ERROR_GRACE_SECONDS = 10     # how long a process may keep running after a traceback appears
UNIQUE_SAMPLE_SECONDS = 0.5  # unique memory is costlier to read than resident memory
BUDGET_SLACK = 1.15          # the evaluator stops itself on a projection; this is the hard stop
BUDGET_EXTRA_SECONDS = 60    # scoring, importance and the model load check sit outside the folds


def attempt_dir(context: RunContext, model: str, attempt: int) -> Path:
    """runs/{run_id}/{model}/attempt_{n} — one directory per attempt.

    Namespaced by model because workers run in parallel and would otherwise
    overwrite each other's candidates and output.
    """
    return Path(context.run_dir) / model / f"attempt_{attempt}"


# ---------------------------------------------------------------------------
# contract and command
# ---------------------------------------------------------------------------

def _excluded(context: RunContext, extra_excluded) -> List[str]:
    """The profile's drop list plus every column excluded since, in a stable order."""
    columns = list(context.drop_columns)
    for column in extra_excluded or ():
        if column not in columns:
            columns.append(column)
    return columns


def write_contract(
    context: RunContext, model: str, out_dir: Path, backend: str, extra_excluded=()
) -> Path:
    """The evaluator's instructions for this attempt, with paths as it will see them."""
    plan = context.split_plan
    train_path = str(Path(context.train_path).resolve())
    folds_path = str(Path(context.folds_path).resolve())
    if backend == "sandbox":
        train_path = f"/data/{Path(context.train_path).name}"
        folds_path = f"/data/{Path(context.folds_path).name}"

    contract = Contract(
        model=model,
        target=context.target,
        task_type=context.task_type,
        metrics=list(context.eval_matrics),
        primary_metric=context.primary_metric,
        train_path=train_path,
        folds_path=folds_path,
        seed=plan.random_seed,
        n_jobs=context.n_jobs,
        time_budget_seconds=context.time_budget_seconds,
        excluded_columns=_excluded(context, extra_excluded),
        columns=list(context.columns),
        ordered=plan.cv_strategy == "time_series_split",
        group_column=plan.group_column if plan.cv_strategy == "group_kfold" else None,
    )
    return contract.save(out_dir / CONTRACT_FILE)


def _environment() -> dict:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _subprocess_command(context: RunContext, out_dir: Path, final: bool, load_check: bool = False) -> list:
    command = [sys.executable, "-m", "automl_runtime.evaluate", "--attempt-dir", str(out_dir.resolve())]
    if final or load_check:
        command += ["--load-check" if load_check else "--final", "--test", str(Path(context.test_path).resolve())]
    return command


def _sandbox_command(context: RunContext, final: bool, load_check: bool = False) -> list:
    command = ["python", "-m", "automl_runtime.evaluate", "--attempt-dir", "/work"]
    if final or load_check:
        command += ["--load-check" if load_check else "--final", "--test", f"/data/{Path(context.test_path).name}"]
    return command


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------

_volumes: Dict[Tuple[str, str], str] = {}
_volumes_lock = threading.Lock()


def _run_volume(client, context: RunContext, kind: str, fresh: bool = False) -> str:
    """The run's data on the sandbox host, uploaded once per run.

    Two volumes, so a loop attempt never sees the test set: "training" holds the
    training file and the folds, "final" adds the test file for the final
    evaluation and the load check.
    """
    key = (context.run_dir, kind)
    with _volumes_lock:
        if fresh or key not in _volumes:
            paths = [context.train_path, context.folds_path] + ([context.test_path] if kind == "final" else [])
            volume = client.create_volume({"run": str(context.run_id), "data": kind}, VOLUME_TTL)
            client.put_volume_files(volume, {Path(path).name: Path(path) for path in paths})
            _volumes[key] = volume
        return _volumes[key]


def release_run(context: RunContext) -> None:
    """Drop the run's data from the sandbox host once the run is over."""
    if context.backend != "sandbox":
        return
    client = hooks.sandbox_for(context.run_id)
    for kind in ("training", "final"):
        with _volumes_lock:
            volume = _volumes.pop((context.run_dir, kind), None)
        if volume:
            try:
                client.delete_volume(volume, retry=False)
            except (SandboxError, SandboxUnavailable) as error:
                print(f"run data left for the sandbox host's reaper: {error}")


def _follow(client, sandbox: str, exec_id: str):
    """Poll an exec to its end -> (final status, stdout, stderr, died_after_error).

    The same rule as _execute: a traceback followed by ERROR_GRACE_SECONDS
    without an exit is a failure, not a slow model, and the sandbox is killed.
    """
    out, err = [], []
    out_at = err_at = 0
    first_error_at = None
    died_after_error = False
    while True:
        status = client.poll(sandbox, exec_id, out_at, err_at)
        out.append(status["stdout"])
        err.append(status["stderr"])
        out_at, err_at = status["stdout_offset"], status["stderr_offset"]
        if status["state"] != "running":
            return status, "".join(out), "".join(err), died_after_error

        now = time.monotonic()
        if first_error_at is None and _has_traceback("".join(err)):
            first_error_at = now
        elif first_error_at is not None and not died_after_error and now - first_error_at > ERROR_GRACE_SECONDS:
            died_after_error = True
            client.kill(sandbox)
        time.sleep(SANDBOX_POLL_SECONDS)


def _sandbox_execute(
    context: RunContext, out_dir: Path, command: list, files: dict, data: str, deadline: float, memory_mb: float, step: str
):
    """Run one evaluator command in a fresh sandbox and bring /work back into out_dir.

    Returns what _execute returns, plus the host's usage report. The sandbox is
    always deleted: a model load check never reuses the sandbox that trained,
    and under gVisor a sandbox that ran out of memory keeps holding it.
    """
    client = hooks.sandbox_for(context.run_id)
    spec = dict(
        image=RUNNER_IMAGE, cpus=context.n_jobs, memory_mb=int(memory_mb), work_mb=WORK_MB,
        ttl_seconds=int(deadline) + 600,
        labels={"run": str(context.run_id), "model": out_dir.parent.name,
                "attempt": out_dir.name.removeprefix("attempt_"), "step": step},
    )
    try:
        sandbox = client.create_sandbox(**spec, volume=_run_volume(client, context, data))
    except SandboxError as error:
        if error.status != 404:
            raise
        # the run's data expired on the host, or the host is new: upload it again
        sandbox = client.create_sandbox(**spec, volume=_run_volume(client, context, data, fresh=True))

    try:
        client.put_files(sandbox, files)
        exec_id = client.start_exec(sandbox, command, deadline)
        status, stdout, stderr, died_after_error = _follow(client, sandbox, exec_id)
        if status["state"] == "exited":
            client.download(sandbox, out_dir)
    except BaseException:
        try:
            client.delete_sandbox(sandbox, retry=False)
        except (SandboxError, SandboxUnavailable):
            pass  # the host's reaper removes it when its TTL runs out
        raise
    usage = client.delete_sandbox(sandbox)

    if status["state"] == "lost":
        raise SandboxUnavailable("the evaluator's output stopped without an exit code; the sandbox host lost it")
    return (
        status["exit_code"], stdout, stderr, status["elapsed_seconds"], usage.get("peak_memory_mb") or 0.0,
        status["state"] == "timeout", died_after_error, bool(status["oom_killed"]), usage,
    )


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

def _tree(process) -> list:
    """The process and its descendants, tolerating ones that exit while listed."""
    try:
        return [process] + process.children(recursive=True)
    except psutil.Error:
        return [process]


def _tree_unique_mb(process) -> float:
    """Memory that belongs to the process tree alone.

    Resident memory counts shared libraries once per process, so a model with a
    pool of worker processes looks far larger than it is: one evaluator with five
    joblib workers read 1,128 MB resident and 900 MB unique. The ceiling is held
    against the unique figure; resident memory is the fallback where it cannot
    be read.
    """
    total = 0
    for member in _tree(process):
        try:
            total += member.memory_full_info().uss
        except psutil.AccessDenied:
            try:
                total += member.memory_info().rss
            except psutil.Error:
                pass
        except psutil.Error:
            pass
    return total / 1e6


def _drain(stream, sink: list) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
    stream.close()


def _execute(command: list, cwd: Path, environment: dict, timeout: float, memory_limit_mb: float = 0.0):
    """Run a command, watching peak memory, and stop waiting once it is doomed.

    Output is drained by reader threads rather than at the end, so a traceback
    can be seen while the process is still alive. That matters because an error
    raised inside joblib's worker processes can leave the parent sitting there:
    an InvalidParameterError that takes milliseconds to hit was costing the
    entire time budget and being reported as a slow model.

    It is also stopped when its process tree outgrows `memory_limit_mb`: a model
    that would take the whole machine down takes only its own attempt instead.

    Returns (exit_code, stdout, stderr, wall_seconds, peak_mb, timed_out, died_after_error, over_memory).
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
    over_memory = False
    first_error_at = None
    unique_checked_at = 0.0
    shared_mb = 0.0              # resident memory last seen to be shared libraries, not the model's own

    while process.poll() is None:
        # the whole tree at once: joblib workers hold memory in children
        current = 0
        for member in _tree(process):
            try:
                current += member.memory_info().rss
            except psutil.Error:
                pass
        peak_bytes = max(peak_bytes, current)

        now = time.perf_counter()
        current_mb = current / 1e6
        if memory_limit_mb and current_mb > memory_limit_mb:
            # resident memory is only a trigger; the decision is made on unique
            # memory. New allocations are the model's own, so growth past what
            # was shared at the last reading is checked at once, not after the
            # sampling interval: a short-lived spike can take the machine down.
            if current_mb - shared_mb > memory_limit_mb or now - unique_checked_at >= UNIQUE_SAMPLE_SECONDS:
                unique_checked_at = now
                unique_mb = _tree_unique_mb(process)
                shared_mb = max(0.0, current_mb - unique_mb)
                if unique_mb > memory_limit_mb:
                    over_memory = True

        # a traceback has appeared but the process is still running. Give it a
        # moment to exit on its own — a library may recover — then stop waiting.
        if first_error_at is None and _has_traceback("".join(err_lines)):
            first_error_at = now
        elif first_error_at is not None and now - first_error_at > ERROR_GRACE_SECONDS:
            died_after_error = True

        if over_memory or died_after_error or now - started > timeout:
            timed_out = not (died_after_error or over_memory)
            for member in reversed(_tree(process)):
                try:
                    member.kill()
                except psutil.Error:
                    pass
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
        over_memory,
    )


_STAGE_MARK = re.compile(r"^\[stage: ([a-z_]+)\]\s*$", re.MULTILINE)


def _stage_tag(stdout: str) -> str:
    """`[stage: name]` for the last stage the evaluator announced, for a process
    stopped before it could say where it failed."""
    marks = _STAGE_MARK.findall(stdout or "")
    return f"[stage: {marks[-1]}]\n" if marks else ""


def _has_traceback(text: str) -> bool:
    return "Traceback (most recent call last)" in text or bool(
        re.search(r"^[A-Za-z_.]*(Error|Exception):", text, re.MULTILINE)
    )


_MISSING_MODULE = re.compile(r"No module named '([^']+)'")
_BAD_NAME = re.compile(r"cannot import name '[^']+' from '([^']+)'")


def _is_dependency_error(text: str, context: RunContext) -> bool:
    """A package that is not installed, as opposed to a wrong import path.

    `cannot import name 'TransformedTargetRegressor' from 'sklearn.preprocessing'`
    names a package that IS installed — the candidate simply looked in the wrong
    module, which the next attempt can fix. Calling that a dependency failure
    would abandon a model over a typo.
    """
    installed = set(context.environment) | {"automl_runtime"}

    match = _MISSING_MODULE.search(text)
    if match:
        return match.group(1).split(".")[0] not in installed

    match = _BAD_NAME.search(text)
    if match:
        return match.group(1).split(".")[0] not in installed

    return False


def _check_model(context: RunContext, out_dir: Path, backend: str, ceiling: float) -> List[str]:
    """Open the saved model in a process of its own, once the final evaluation has exited.

    Inside the evaluator the check was a second interpreter held against the
    ceiling of the one that had just fitted the model, and stopping it lost test
    scores that were already computed. Here the ceiling is held against loading
    the model alone, and the scores stand whatever the check finds.
    """
    result_path = out_dir / LOAD_CHECK_FILE
    result_path.unlink(missing_ok=True)

    if backend == "sandbox":
        # the pickle refers to what the candidate defined, so the module goes with it
        files = {name: out_dir / name for name in (CANDIDATE_FILE, CONTRACT_FILE, MODEL_FILE)}
        exit_code, stdout, stderr, _, _, timed_out, _, over_memory, _ = _sandbox_execute(
            context, out_dir, _sandbox_command(context, False, load_check=True), files, "final",
            LOAD_CHECK_TIMEOUT, ceiling, "load_check",
        )
    else:
        command = _subprocess_command(context, out_dir, False, load_check=True)
        exit_code, stdout, stderr, _, _, timed_out, _, over_memory = _execute(
            command, out_dir, _environment(), LOAD_CHECK_TIMEOUT, ceiling
        )
    (out_dir / "stdout_load_check.txt").write_text(stdout, encoding="utf-8")
    (out_dir / "stderr_load_check.txt").write_text(stderr, encoding="utf-8")

    if over_memory:
        return [
            f"{MODEL_FILE} could not be verified: opening it in a fresh process needed more than the "
            f"{ceiling:.0f} MB an evaluator may use on this machine. The test scores stand; loading the "
            "model may need a larger machine"
        ]
    if timed_out:
        problem = f"loading it took longer than {LOAD_CHECK_TIMEOUT}s"
    elif result_path.exists():
        try:
            checked = json.loads(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            checked = {"status": "error", "error": f"unreadable {LOAD_CHECK_FILE}: {error}"}
        if checked.get("status") == "ok":
            return []
        problem = checked.get("error") or "unknown error"
    else:
        tail = stderr.strip().splitlines()
        problem = tail[-1][:240] if tail else f"the check exited with code {exit_code}"
    return [f"{MODEL_FILE} cannot be loaded in a fresh process ({problem}); it is not usable as a deliverable"]


# ---------------------------------------------------------------------------
# checks before a process is spent
# ---------------------------------------------------------------------------

def _imported_modules(code: str) -> set:
    """Top-level module names the candidate imports, by parsing rather than running."""
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
    """Parse the candidate without running it."""
    try:
        ast.parse(code)
    except SyntaxError as error:
        where = f"line {error.lineno}" + (f", column {error.offset}" if error.offset else "")
        offending = (error.text or "").rstrip()
        detail = f"{NEWLINE}    {offending}" if offending else ""
        return f"the candidate does not parse: {error.msg} ({where}){detail}"
    return None


def _defines_build(code: str) -> bool:
    tree = ast.parse(code)
    return any(isinstance(node, ast.FunctionDef) and node.name == BUILD_FUNCTION for node in tree.body)


def preflight(code: str, context: RunContext) -> Optional[str]:
    """Reject a candidate that cannot run, before spending a process on it.

    Three instant checks: does it parse, does it define build_pipeline at the
    top level, and does it import anything this environment does not have.
    """
    broken = check_syntax(code)
    if broken is not None:
        return broken

    if not _defines_build(code):
        return (
            f"the candidate defines no top-level {BUILD_FUNCTION} function; the harness calls "
            f"{BUILD_FUNCTION}(columns, task, ctx)"
        )

    if not context.environment:
        return None

    known = set(context.environment) | set(_sys.stdlib_module_names) | {"__future__", "automl_runtime"}
    unknown = sorted(module for module in _imported_modules(code) if module not in known)
    if unknown:
        return (
            f"imports not available in this environment: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(context.environment))}, automl_runtime."
        )

    return _unresolved_import(code, context)


_LOCATIONS: dict = {}


def _where_defined(package: str, name: str) -> List[str]:
    """Public modules of an installed package that define `name`, found by looking.

    Nothing is hard-coded: a name that moved between library versions is found
    wherever the installed version keeps it.
    """
    key = (package, name)
    if key in _LOCATIONS:
        return _LOCATIONS[key]

    found: List[str] = []
    try:
        root = importlib.import_module(package)
        paths = getattr(root, "__path__", None) or []
        for info in pkgutil.iter_modules(paths, prefix=f"{package}."):
            leaf = info.name.rsplit(".", 1)[-1]
            if leaf.startswith("_") or leaf in ("tests", "externals", "experimental", "conftest"):
                continue
            try:
                module = importlib.import_module(info.name)
            except Exception:
                continue
            if name in getattr(module, "__all__", ()) or hasattr(module, name):
                found.append(info.name)
    except Exception:
        pass

    _LOCATIONS[key] = found[:3]
    return _LOCATIONS[key]


def _unresolved_import(code: str, context: RunContext) -> Optional[str]:
    """A name imported from a module that does not have it, caught without running.

    Checked only for installed third-party packages the environment vetted, and
    only when candidates run in this interpreter's environment: resolving a name
    means importing the module, and a container's libraries are not these.
    """
    if context.backend != "subprocess":
        return None
    vetted = set(context.environment) | {"automl_runtime"}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in vetted and "." in alias.name:
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as error:
                        return f"cannot import {alias.name}: {error}"
            continue
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
            continue
        package = node.module.split(".")[0]
        if package not in vetted:
            continue
        try:
            module = importlib.import_module(node.module)
        except ImportError as error:
            return f"cannot import {node.module}: {error}"
        for alias in node.names:
            if alias.name == "*" or hasattr(module, alias.name):
                continue
            try:
                importlib.import_module(f"{node.module}.{alias.name}")
                continue
            except ImportError:
                pass
            places = [place for place in _where_defined(package, alias.name) if place != node.module]
            hint = f"; it is in {' or '.join(places)}" if places else f"; {package} has no public {alias.name}"
            return f"cannot import name '{alias.name}' from '{node.module}'{hint}"
    return None


def _cached_record(
    out_dir: Path,
    digest: str,
    attempt: int,
    prior_attempts: List[int],
    generation: int,
    kind: str,
) -> Optional[AttemptRecord]:
    """Reuse the result of an identical candidate run earlier in THIS loop.

    A retry that changes nothing is common — the judge asks for something the
    generator already did — and re-evaluating it teaches nobody anything.

    Only the attempts the caller says it has already run are considered. Reading
    whatever happens to be on disk would match this attempt's own directory, and
    would reuse records left by an entirely different run.
    """
    for number in sorted(prior_attempts):
        sibling = out_dir.parent / f"attempt_{number}"
        if sibling == out_dir:
            continue
        hash_file = sibling / "candidate.sha256"
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
                "script_path": str(out_dir / CANDIDATE_FILE),
                "changes": f"identical to attempt {record.attempt}; not re-run",
                "cached_from": record.cached_from or record.attempt,
                "usage": {},
            }
        )
    return None


# ---------------------------------------------------------------------------
# reading the evaluator's result
# ---------------------------------------------------------------------------

def _apply_result(record: AttemptRecord, result: dict, context: RunContext) -> None:
    status = result.get("status", "error")
    stage = result.get("stage", "")
    error = result.get("error", "")

    record.warnings = list(result.get("warnings") or [])
    record.excluded_columns = list(result.get("excluded_columns") or [])
    record.fit_seconds = result.get("fit_seconds")
    record.artifacts = {k: str(v) for k, v in (result.get("artifacts") or {}).items()}

    if status == "ok":
        record.cv_scores = {k: float(v) for k, v in (result.get("cv_scores") or {}).items() if v is not None}
        record.test_scores = {k: float(v) for k, v in (result.get("test_scores") or {}).items() if v is not None}
        record.diagnostics = {k: float(v) for k, v in (result.get("diagnostics") or {}).items() if v is not None}
        record.fold_scores = list(result.get("fold_scores") or [])
        return

    if status in ("timeout", "out_of_memory"):
        record.status = status
    elif stage == "harness":
        record.status = "error"
    else:
        record.status = "dependency_error" if _is_dependency_error(error, context) else "error"

    prefix = (
        "[harness error: a fault in the evaluator, not in candidate.py]"
        if stage == "harness"
        else f"[stage: {stage}]"
    )
    record.traceback = f"{prefix}\n{error}"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run_candidate(
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
    extra_excluded: Optional[List[str]] = None,
    ablation_of: Optional[int] = None,
) -> AttemptRecord:
    # before every candidate run, the loop's, a leak check's or the final one
    hooks.check_stop(context.run_id)
    out_dir = attempt_dir(context, model, attempt)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = out_dir / CANDIDATE_FILE
    candidate_path.write_text(code, encoding="utf-8")
    # the final run re-uses the loop attempt's directory; its record must not
    # replace the one holding that attempt's cross-validation
    record_file = out_dir / ("record_final.json" if final else "record.json")
    # the same module on a different set of columns is a different experiment
    excluded = _excluded(context, extra_excluded)
    fingerprint = code + NEWLINE + "# excluded: " + ",".join(sorted(excluded))
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    (out_dir / "candidate.sha256").write_text(digest, encoding="utf-8")

    if not final and prior_attempts:
        cached = _cached_record(out_dir, digest, attempt, prior_attempts, generation, kind)
        if cached is not None:
            record_file.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
            return cached

    record = AttemptRecord(
        model=model,
        attempt=attempt,
        generation=generation,
        kind=kind,
        script_path=str(candidate_path),
        changes=changes,
        excluded_columns=excluded,
        ablation_of=ablation_of,
    )

    blocked = preflight(code, context)
    if blocked is not None:
        record.status = "syntax_error" if blocked.startswith("the candidate does not parse") else (
            "dependency_error" if blocked.startswith("imports not available") else "error"
        )
        record.traceback = f"[stage: preflight]\n{blocked}"
        record_file.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        print(f"  preflight: {blocked}")
        return record

    write_contract(context, model, out_dir, backend, extra_excluded)
    result_path = out_dir / (FINAL_RESULT_FILE if final else RESULT_FILE)
    result_path.unlink(missing_ok=True)
    hard_limit = context.time_budget_seconds * BUDGET_SLACK + BUDGET_EXTRA_SECONDS

    if backend == "sandbox":
        # the sandbox's memory limit is the ceiling, and the kernel holds it; the
        # host queues the attempt while its memory budget is spoken for
        ceiling = context.worker_memory_mb
        command = _sandbox_command(context, final)
        files = {CANDIDATE_FILE: candidate_path, CONTRACT_FILE: out_dir / CONTRACT_FILE}
        exit_code, stdout, stderr, wall_seconds, peak_mb, timed_out, died_after_error, over_memory, usage = (
            _sandbox_execute(
                context, out_dir, command, files, "final" if final else "training",
                hard_limit, ceiling, "final" if final else "loop",
            )
        )
        record.usage = usage
    elif backend == "subprocess":
        command = _subprocess_command(context, out_dir, final)
        # other applications may have taken memory since the run was planned
        waited = wait_for_memory(context.worker_memory_mb)
        if waited >= 1:
            print(f"  waited {waited:.0f}s for {context.worker_memory_mb:.0f} MB of free memory")
        ceiling = memory_limit(context.worker_memory_mb, context.max_concurrency) if context.worker_memory_mb else 0.0
        # run from the attempt directory: libraries that write training logs into
        # the working directory (catboost_info) then leave them with the attempt
        exit_code, stdout, stderr, wall_seconds, peak_mb, timed_out, died_after_error, over_memory = _execute(
            command, out_dir, _environment(), hard_limit, ceiling
        )
    else:
        raise ValueError(f"unknown backend: {backend}")

    suffix = "_final" if final else ""
    (out_dir / f"stdout{suffix}.txt").write_text(stdout, encoding="utf-8")
    (out_dir / f"stderr{suffix}.txt").write_text(stderr, encoding="utf-8")
    (out_dir / f"command{suffix}.txt").write_text(shlex.join(command), encoding="utf-8")

    record.stdout = stdout[-STDOUT_KEPT:]
    record.wall_seconds = round(wall_seconds, 3)
    record.peak_memory_mb = round(peak_mb, 1)
    # where a process that never wrote its result had got to
    reached = _stage_tag(stdout)

    if over_memory:
        # a resource failure, not a bug: it goes to the judge, who can make the
        # model smaller, not to the fixer
        record.status = "out_of_memory"
        record.traceback = reached + (
            f"stopped when its processes held more than the {ceiling:.0f} MB this evaluator may use on "
            "this machine. Make the model smaller rather than asking for more: fewer or shallower trees "
            "(max_depth, min_samples_leaf, max_leaf_nodes), fewer estimators, a subsample, or a cheaper "
            "encoding of wide categoricals."
        )
    elif result_path.exists():
        # the evaluator finished and explained itself, whatever the candidate did
        try:
            _apply_result(record, json.loads(result_path.read_text(encoding="utf-8")), context)
        except (ValueError, OSError) as error:
            record.status = "error"
            record.traceback = f"[harness error: unreadable {result_path.name}] {error}"
    elif died_after_error:
        # it raised and then sat there, typically a joblib worker taking the
        # parent down with it. The error is the result.
        record.status = "dependency_error" if _is_dependency_error(stderr, context) else "error"
        record.traceback = reached + (
            f"raised after {record.wall_seconds}s and did not exit, so it was stopped:\n"
            + stderr[-STDOUT_KEPT:]
        )
    elif timed_out:
        record.status = "timeout"
        note = f"{reached}killed after {hard_limit:.0f}s (time budget {context.time_budget_seconds}s)"
        if _has_traceback(stderr):
            note += (
                "\nthe process also reported errors before it was stopped, so this may be a "
                "failure rather than an expensive fit:\n" + stderr[-STDOUT_KEPT:]
            )
        record.traceback = note
    else:
        # no result at all: the interpreter itself died (out of memory, a crash
        # in native code) before the evaluator could write anything
        record.status = "error"
        record.traceback = reached + (
            f"the evaluator exited with code {exit_code} without writing {result_path.name}; "
            f"peak memory {record.peak_memory_mb:.0f} MB.\n" + stderr[-STDOUT_KEPT:]
        )

    if final and record.status == "ok" and "model" in record.artifacts:
        record.warnings += _check_model(context, out_dir, backend, ceiling)

    if record.status == "ok" and not final:
        # a clean evaluation is not the same as a trustworthy one. Final runs
        # are checked in final_eval, which holds the loop scores they need.
        record.findings = check_attempt(record, context)
        record.warnings += [finding.message for finding in record.findings]
        for finding in record.findings:
            print(f"  guard [{finding.severity}]: {finding.message}")
            hooks.emit(context.run_id, "guard_finding", model=model, attempt=attempt, finding=finding.kind,
                       severity=finding.severity, message=finding.message)

    record_file.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record
