"""
environment.py
--------------
Find out what the backend can actually import, optionally install what is
missing, and freeze the answer for the run.

Discovery is done by asking the environment, not by reading a list someone
wrote months ago. What IS fixed is the allowlist: the package names come from a
model list an LLM produced, and `pip install <whatever the model said>` is a
typosquatting install waiting to happen. A hallucinated `catboost-gpu` is a
plausible name that may well resolve to something real owned by someone else.
So discovery is open and installation is bounded.

Whatever comes out of here is pinned for the whole run. Installing a package
midway through would mean the attempts before it and the attempts after it were
measured under different conditions, which is exactly the variable the frozen
split and the shared folds exist to remove.

    probe_environment(backend, client)         -> {import_name: version}
    resolve_models(models, environment)        -> (available, unavailable)
    provision(models, environment, policy)     -> (installed, notes)
    runtime_hash()                             -> the hash of automl_runtime's source

A sandbox host reports its image's packages in GET /v1/info, with the hash of
the automl_runtime baked into it. A host whose hash differs from ours would
score models with a different evaluator, so it is refused.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# model name -> (import name, pip name). Anything absent needs only sklearn.
MODEL_PACKAGES = {
    "xgboost": ("xgboost", "xgboost"),
    "lightgbm": ("lightgbm", "lightgbm"),
    "catboost": ("catboost", "catboost"),
}

CORE_IMPORTS = ["sklearn", "pandas", "numpy", "scipy", "joblib"]

# every model the config generator may choose from, so availability can be
# stated to it up front rather than pruned afterwards
ALL_MODELS = [
    "logistic_regression", "linear_regression", "ridge", "lasso", "elastic_net",
    "decision_tree", "random_forest", "extra_trees", "gradient_boosting",
    "xgboost", "lightgbm", "catboost", "svm", "knn", "naive_bayes", "mlp",
]

PROVISION_NEVER = "never"
PROVISION_ONCE = "once"
PROVISION_IMAGE = "image"

RUNNER_IMAGE = "automl-runner"
RUNTIME_DIR = Path(__file__).resolve().parents[2] / "automl_runtime"
# import names whose distribution is called something else
DISTRIBUTIONS = {"sklearn": "scikit-learn"}


def required_import(model: str) -> str:
    return MODEL_PACKAGES.get(model, ("sklearn", "scikit-learn"))[0]


def available_model_names(environment: Dict[str, str]) -> List[str]:
    return [model for model in ALL_MODELS if required_import(model) in environment]


def _probe_names() -> List[str]:
    return CORE_IMPORTS + [import_name for import_name, _ in MODEL_PACKAGES.values()]


def _probe_here(names: List[str]) -> Dict[str, str]:
    found = {}
    for name in names:
        if importlib.util.find_spec(name) is None:
            continue
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # installed but the distribution name differs from the import name
            try:
                found[name] = getattr(importlib.import_module(name), "__version__", "unknown")
            except Exception:
                found[name] = "unknown"
    return found


def runtime_hash(root: Path = RUNTIME_DIR) -> str:
    """Same as sandbox/scripts/runtime_hash.py, which labels the runner image."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).glob("*.py")):
        digest.update(path.name.encode() + b"\0")
        # CRLF -> LF so a Windows checkout hashes the same as Linux
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


def _normal(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _probe_sandbox(client, names: List[str]) -> Dict[str, str]:
    """Ask the sandbox host what its runner image has, and refuse a different runtime."""
    image = client.info()["images"].get(RUNNER_IMAGE) or {}
    if image.get("error") or "packages" not in image:
        raise RuntimeError(f"the sandbox host has no usable {RUNNER_IMAGE} image: {image.get('error', 'no package list')}")
    ours, theirs = runtime_hash(), image.get("runtime_hash")
    if theirs != ours:
        raise RuntimeError(
            f"the sandbox host runs automl_runtime {theirs}, this code is {ours}: rebuild its runner image "
            "with sandbox/scripts/build_runner_image.sh"
        )
    packages = {_normal(name): version for name, version in image["packages"].items()}
    found = {}
    for name in names:
        version = packages.get(_normal(DISTRIBUTIONS.get(name, name)))
        if version:
            found[name] = version
    return found


def probe_environment(backend: str = "subprocess", client=None) -> Dict[str, str]:
    names = _probe_names()
    if backend == "sandbox":
        return _probe_sandbox(client, names)
    return _probe_here(names)


def resolve_models(models: List[str], environment: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    """Split a model list into what this environment can run and what it cannot."""
    available, unavailable = [], {}
    for model in models:
        import_name = required_import(model)
        if import_name in environment:
            available.append(model)
        else:
            unavailable[model] = f"{import_name} is not installed in this environment"
    return available, unavailable


def provision(
    models: List[str],
    environment: Dict[str, str],
    policy: str = PROVISION_NEVER,
    backend: str = "subprocess",
) -> Tuple[List[str], List[str]]:
    """Install the missing packages a model list needs, once, before any training.

    Only packages named in MODEL_PACKAGES are ever installed, and only the pip
    name this file holds — never a string that came from the model list.
    """
    notes: List[str] = []

    missing = {
        model: MODEL_PACKAGES[model]
        for model in models
        if model in MODEL_PACKAGES and MODEL_PACKAGES[model][0] not in environment
    }
    if not missing:
        return [], notes

    names = ", ".join(sorted(missing))
    if policy == PROVISION_NEVER:
        notes.append(f"missing packages for {names}; provisioning is off, so those models are dropped")
        return [], notes

    if backend == "sandbox":
        # installing inside a sandbox would need the network it deliberately lacks
        notes.append(
            f"missing packages for {names} in the runner image; rebuild it with those in "
            "runner_requirements.txt rather than installing at run time"
        )
        return [], notes

    if policy == PROVISION_IMAGE:
        notes.append(f"missing packages for {names}; policy is 'image', so no run-time install")
        return [], notes

    installed: List[str] = []
    for model, (import_name, pip_name) in sorted(missing.items()):
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", pip_name],
            capture_output=True, text=True,
        )
        if completed.returncode == 0:
            installed.append(pip_name)
            notes.append(f"installed {pip_name} for {model}")
        else:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            notes.append(f"could not install {pip_name} for {model}: {tail[-1] if tail else 'pip failed'}")

    return installed, notes
