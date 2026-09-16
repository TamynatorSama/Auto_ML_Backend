"""
resume.py
---------
Pick a run up from its directory.

Planning writes context.json and config.json into the run directory, and every
model writes result.json there when it finishes. A run that crashed, or was
stopped, is resumed by loading the first two and invoking the model graph
again: finished models are read back, and only the rest are trained.

    load_run(run_dir)   -> (context, config) or None
    resume_run(run_dir) -> the model graph's final state
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from models import Configs, RunContext
from utils.reusable.resources import plan_resources


def load_run(run_dir: str | Path) -> Optional[Tuple[RunContext, Configs]]:
    run_dir = Path(run_dir)
    context_path, config_path = run_dir / "context.json", run_dir / "config.json"
    if not context_path.exists() or not config_path.exists():
        return None
    context = RunContext.model_validate_json(context_path.read_text(encoding="utf-8"))
    config = Configs.model_validate_json(config_path.read_text(encoding="utf-8"))

    # the machine may have more or less room than when the run was planned, and
    # a run resumed after running out of memory must not repeat the plan that
    # ran out. n_jobs is only ever lowered: a pool of worker processes is paid
    # for in memory, and seeded estimators give the same model either way.
    plan = plan_resources(context.train_path, len(config.models))
    override = os.environ.get("AUTOML_CONCURRENCY")
    concurrency = int(override) if override else plan.max_concurrency
    context = context.model_copy(update={
        "max_concurrency": concurrency,
        "n_jobs": min(context.n_jobs, plan.n_jobs),
        "worker_memory_mb": plan.worker_memory_mb,
        "resource_plan": f"re-planned on resume: {plan.reason}",
    })
    return context, config


def resume_run(run_dir: str | Path, topic: str = ""):
    from code_gen_eval.base import app as code_gen_eval_app

    loaded = load_run(run_dir)
    if loaded is None:
        raise FileNotFoundError(f"{run_dir} has no context.json and config.json to resume from")
    context, config = loaded
    return code_gen_eval_app.invoke({"topic": topic, "context": context, "config": config})
