"""
candidate.py
------------
Import a generated candidate module, call its interface, and open saved models.

A candidate is imported as the module `candidate` from its attempt directory,
never run as `__main__`. That is what makes the models it produces loadable
anywhere: pickle stores a class or function by module path, so a transformer
the candidate defines is saved as `candidate.MyTransformer` and loads again in
any process that can import `candidate.py` from the same directory.

    import_candidate(directory)            -> module
    call_build(module, columns, task, ctx) -> unfitted estimator
    call_fit_params(module, ctx)           -> dict of fit keyword arguments
    load_model(path)                       -> fitted estimator
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict

MODULE_NAME = "candidate"
FILENAME = "candidate.py"


class InterfaceError(Exception):
    """The module does not expose what the harness calls."""


def _prepend_path(directory: Path) -> None:
    text = str(directory.resolve())
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def import_candidate(directory: str | Path):
    directory = Path(directory)
    if not (directory / FILENAME).exists():
        raise InterfaceError(f"no {FILENAME} in {directory}")
    # no __pycache__ in attempt directories: they are records, not packages
    sys.dont_write_bytecode = True
    _prepend_path(directory)
    sys.modules.pop(MODULE_NAME, None)
    importlib.invalidate_caches()
    return importlib.import_module(MODULE_NAME)


def _accepted_positional(function) -> int:
    parameters = inspect.signature(function).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return 99
    return sum(
        1 for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def call_build(module, columns, task: str, ctx) -> Any:
    build = getattr(module, "build_pipeline", None)
    if not callable(build):
        raise InterfaceError(
            "candidate.py defines no build_pipeline function. The harness calls "
            "build_pipeline(columns, task, ctx) and cross-validates the estimator it returns."
        )
    # tolerate a candidate that ignores ctx or task: those are its choice, not a fault
    arguments = (columns, task, ctx)[: max(1, _accepted_positional(build))]
    return build(*arguments)


def call_fit_params(module, ctx) -> Dict[str, Any]:
    fit_params = getattr(module, "fit_params", None)
    if fit_params is None:
        return {}
    if not callable(fit_params):
        raise InterfaceError("fit_params is defined but is not a function")
    params = fit_params(ctx)
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise InterfaceError(f"fit_params(ctx) must return a dict, got {type(params).__name__}")
    return params


def wants_fit_params(module) -> bool:
    return callable(getattr(module, "fit_params", None))


def load_model(path: str | Path):
    """Open a model the evaluator saved, from any process.

    The candidate module next to the model is made importable first, because the
    pickle refers to anything the candidate defined by `candidate.<name>`.
    """
    import joblib

    path = Path(path)
    sys.dont_write_bytecode = True
    _prepend_path(path.parent)
    sys.modules.pop(MODULE_NAME, None)
    return joblib.load(path)
