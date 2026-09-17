"""
report.py
---------
The shape of a finished run, as one JSON document.

This is the contract an interface is built against, so it is written for a
consumer that has no access to the run: every number a view might show is a
field, nothing has to be recomputed from prose, and anything a chart needs to
colour or sort by — a metric's direction, a warning's severity — is stated
rather than implied.

`schema_version` changes when a field is removed or its meaning changes. Adding
a field does not bump it.
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from .results import INF_SAFE

SCHEMA_VERSION = "1.0"


class MetricSpec(BaseModel):
    """What a metric is, so a view can rank and colour it without a lookup table."""

    name: str
    direction: str              # "lower" | "higher" — which way is better
    primary: bool = False
    drives_improvement: bool = False    # the one the loop's stopping rule watches


class ScoreRow(BaseModel):
    """One model, as a row of the comparison table."""

    model_config = INF_SAFE

    rank: Optional[int] = None
    model: str
    status: str                 # max_tries | no_improvement | failed | unavailable | ...
    selected: bool = False
    test_scores: Dict[str, float] = {}
    cv_scores: Dict[str, float] = {}
    fold_scores: List[Optional[float]] = []  # primary metric by fold, for the cv score above
    generations: int = 0        # modelling attempts
    repairs: int = 0            # fixes that did not spend a modelling attempt
    executions: int = 0         # every script run, generations and repairs together
    best_attempt: Optional[int] = None
    wall_seconds: float = 0.0
    artifacts: Dict[str, str] = {}
    note: str = ""              # why there is no score, when there is none
    eligibility: str = "clean"  # clean | unverified | blocked


class GenerationStep(BaseModel):
    """One modelling attempt: what it changed, and what that bought."""

    model_config = INF_SAFE

    generation: int
    attempt: int
    status: str
    changes: str = ""
    score: Optional[float] = None
    delta: Optional[float] = None       # improvement over the best before it, signed so
                                        # positive is always better
    repairs: int = 0
    judge_notes: str = ""


class FeatureWeight(BaseModel):
    feature: str
    importance: float
    share: float = 0.0          # of the total, 0-1, so a bar needs no client maths


class ErrorBand(BaseModel):
    """Error inside one slice of the target range. Regression only."""

    model_config = INF_SAFE

    band: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    rows: int
    mean_absolute_error: float
    mean_signed_error: float    # positive over-predicts, negative under-predicts


class ConfusionCell(BaseModel):
    """Classification only."""

    actual: str
    predicted: str
    rows: int


class Warning(BaseModel):
    """Something raised about a result, with enough to render it without parsing."""

    model: str
    severity: str               # "critical" | "warning"
    stage: str                  # "cross_validation" | "test" | "artifact" | "run"
    message: str


class DatasetInfo(BaseModel):
    model_config = INF_SAFE

    target: str
    task_type: str
    train_rows: int = 0
    test_rows: int = 0
    train_path: str = ""
    test_path: str = ""
    target_stats: Dict[str, float] = {}


class Protocol(BaseModel):
    """How the run was measured. Without this the scores are unverifiable."""

    split_method: str = ""
    test_size: float = 0.0
    seed: int = 0
    drop_duplicates: bool = False
    cv_strategy: str = ""
    cv_folds: int = 0
    split_reason: str = ""
    split_warnings: List[str] = []
    baseline_strategy: str = ""
    baseline_applicable: bool = True
    baseline_scores: Dict[str, float] = {}
    max_tries: int = 0
    early_stopping_patience: int = 0
    improvement_delta: float = 0.0
    improvement_mode: str = ""
    time_budget_seconds: int = 0
    environment: Dict[str, str] = {}
    requirements: List[Dict[str, str]] = []


class RunTotals(BaseModel):
    models_planned: int = 0
    models_scored: int = 0
    executions: int = 0
    generations: int = 0
    repairs: int = 0
    failures: int = 0
    wall_seconds: float = 0.0


class RunReport(BaseModel):
    model_config = INF_SAFE

    schema_version: str = SCHEMA_VERSION
    run_id: int
    generated_at: str
    status: str                 # "complete" | "no model produced a score"

    dataset: DatasetInfo
    protocol: Protocol
    totals: RunTotals
    metrics: List[MetricSpec] = []

    selected_model: Optional[str] = None
    selection_reason: str = ""
    improvement_over_baseline: Optional[float] = None    # fraction, positive is better

    comparison: List[ScoreRow] = []
    trace: Dict[str, List[GenerationStep]] = {}
    importance: List[FeatureWeight] = []
    error_bands: List[ErrorBand] = []
    confusion: List[ConfusionCell] = []
    warnings: List[Warning] = []
    # columns removed during the run, with the evidence that removed them
    exclusions: List[Dict[str, Any]] = []
    # what the pre-fit relation screen found, including relations kept because
    # the schema declares their columns known at prediction time
    leakage_screen: List[Dict[str, Any]] = []

    narrative: str = ""
