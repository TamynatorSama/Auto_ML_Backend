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

    probe_environment(backend)                 -> {import_name: version}
    resolve_models(models, environment)        -> (available, unavailable)
    provision(models, environment, policy)     -> (installed, notes)
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
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


def _probe_docker(image: str, names: List[str]) -> Dict[str, str]:
    """Ask the image itself what it has, rather than trusting a requirements file."""
    script = (
        "import importlib.util, importlib.metadata, json\n"
        f"names = {names!r}\n"
        "found = {}\n"
        "for n in names:\n"
        "    if importlib.util.find_spec(n) is None: continue\n"
        "    try: found[n] = importlib.metadata.version(n)\n"
        "    except Exception: found[n] = 'unknown'\n"
        "print(json.dumps(found))\n"
    )
    try:
        completed = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", image, "python", "-c", script],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}


def probe_environment(backend: str = "subprocess", image: str = "automl-runner") -> Dict[str, str]:
    names = _probe_names()
    if backend == "docker":
        return _probe_docker(image, names)
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

    if backend == "docker":
        # installing inside the training container would need the network that
        # --network none deliberately removes
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
