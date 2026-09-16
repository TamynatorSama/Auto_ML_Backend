from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional

# A diverged model really does score inf, and the default serialiser writes that
# as null, which then fails to load back — so the record of the worst failures
# was the one record that could not be read.
INF_SAFE = ConfigDict(ser_json_inf_nan="constants")


class BaselineResult(BaseModel):
    model_config = INF_SAFE
    strategy: str
    applicable: bool
    cv_scores: Dict[str, float] = {}
    note: str = ""


class PreprocessingRequirement(BaseModel):
    """Preprocessing a script must do, derived from the profile rather than chosen.

    Which models it binds is a property of the algorithm, not of the dataset:
    trees split on order and ignore scale, linear models do not.
    """

    column: Optional[str] = None
    issue: str
    requirement: str
    applies_to: List[str] = []      # model names, or ["*"] for every model


class AttemptRecord(BaseModel):
    model_config = INF_SAFE
    model: str
    attempt: int              # every execution of this model, in order
    generation: int = 1       # the modelling attempt this belongs to
    kind: str = "generate"    # generate | repair
    script_path: str
    status: str = "ok"          # ok | error | timeout | cached
    changes: str = ""
    cv_scores: Dict[str, float] = {}
    test_scores: Dict[str, float] = {}
    fit_seconds: Optional[float] = None      # reported by the script
    wall_seconds: Optional[float] = None     # measured by the runner
    peak_memory_mb: Optional[float] = None
    artifacts: Dict[str, str] = {}
    warnings: List[str] = []
    stdout: str = ""
    traceback: str = ""
    judge_notes: str = ""
    decision: str = ""


class ModelResult(BaseModel):
    model_config = INF_SAFE
    model: str
    status: str
    attempts: List[AttemptRecord] = []
    best_attempt: Optional[int] = None
    best_cv_scores: Dict[str, float] = {}
    test_scores: Dict[str, float] = {}
    # raised while scoring the test set, so they cannot live on a loop attempt
    test_warnings: List[str] = []
    error: str = ""
