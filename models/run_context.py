from pydantic import BaseModel
from typing import Any, Dict, List

from .results import INF_SAFE, BaselineResult, PreprocessingRequirement
from .split_plan import SplitPlan


class RunContext(BaseModel):
    """Everything a code generation worker needs, frozen before the fan-out.

    One object rather than a dozen loose state keys: every worker reads the
    identical bundle, which is what makes their scores comparable at fan-in.
    Workers never write to it.
    """

    model_config = INF_SAFE

    run_id: int
    run_dir: str

    train_path: str
    test_path: str
    split_plan: SplitPlan
    # fold row indices computed once for the whole run, so every model and the
    # baseline are scored on exactly the same rows
    folds_path: str = ""
    # what each candidate is told about the columns it receives (name, kind,
    # dtype, n_unique, missing_pct), taken from the training-set profile
    columns: List[Dict[str, Any]] = []
    summary: str
    target: str
    task_type: str
    target_stats: Dict[str, float] = {}
    drop_columns: List[str] = []
    # hygiene the profile says is required, not preferences
    preprocessing_requirements: List[PreprocessingRequirement] = []

    baseline: BaselineResult

    # columns the schema declares known at prediction time; a leak check never
    # excludes these, whatever it finds
    available_at_prediction: List[str] = []
    # exclude: a confirmed leak on an undeclared column is removed run-wide
    # keep:    it is reported but stays in
    leakage_policy: str = "exclude"
    # relations the pre-fit screen found between columns and the target, with
    # what the policy did about each
    leakage_screen: List[Dict[str, Any]] = []

    eval_matrics: List[str]
    primary_metric: str
    metric_direction: str
    improvement_metric: str
    improvement_delta: float
    improvement_mode: str
    early_stopping_patience: int
    max_tries: int

    time_budget_seconds: int
    n_jobs: int
    # how many models train at once, and what one worker is expected to need;
    # planned from the machine and the data when the run starts
    max_concurrency: int = 2
    # the ceiling a sandbox is killed above, and what the worker's pool counts
    # against the host while it runs — see utils/reusable/resources.py
    worker_memory_mb: float = 0.0
    reserve_memory_mb: float = 0.0
    reserve_cpus: float = 0.0
    resource_plan: str = ""
    plan_fingerprint: str = ""

    # resolved once before the fan-out and pinned, so every model in the run is
    # measured under identical library versions
    backend: str = "subprocess"
    environment: Dict[str, str] = {}
    unavailable_models: Dict[str, str] = {}
    environment_notes: List[str] = []
