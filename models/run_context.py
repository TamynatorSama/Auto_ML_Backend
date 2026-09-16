from pydantic import BaseModel
from typing import Dict, List

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
    summary: str
    target: str
    task_type: str
    target_stats: Dict[str, float] = {}
    drop_columns: List[str] = []
    # hygiene the profile says is required, not preferences
    preprocessing_requirements: List[PreprocessingRequirement] = []

    baseline: BaselineResult

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

    # resolved once before the fan-out and pinned, so every model in the run is
    # measured under identical library versions
    backend: str = "subprocess"
    environment: Dict[str, str] = {}
    unavailable_models: Dict[str, str] = {}
    environment_notes: List[str] = []
