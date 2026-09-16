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


class Finding(BaseModel):
    """Something a check concluded about an attempt, in a form the selection rules can act on."""

    kind: str           # constant_predictions | untransformed_target | missing_predictions |
                        # no_primary_score | worse_than_baseline | suspect_leakage
    severity: str       # blocking: cannot be selected | suspect: needs a leak check | warning
    message: str
    evidence: Dict[str, float] = {}


class AttemptRecord(BaseModel):
    model_config = INF_SAFE
    model: str
    attempt: int              # every execution of this model, in order
    generation: int = 1       # the modelling attempt this belongs to
    kind: str = "generate"    # generate | repair | ablation
    script_path: str          # the candidate module that ran
    status: str = "ok"          # ok | error | syntax_error | dependency_error | timeout | out_of_memory | cached
    changes: str = ""
    # computed by the harness from the pooled out-of-fold predictions; the
    # candidate never reports its own
    cv_scores: Dict[str, float] = {}
    test_scores: Dict[str, float] = {}
    fold_scores: List[Optional[float]] = []  # primary metric, fold by fold
    diagnostics: Dict[str, float] = {}       # scale-free readings: r2, or roc_auc
    fit_seconds: Optional[float] = None      # fitting only, summed over folds
    wall_seconds: Optional[float] = None     # measured by the runner
    peak_memory_mb: Optional[float] = None
    artifacts: Dict[str, str] = {}
    findings: List[Finding] = []
    warnings: List[str] = []                 # the findings and evaluator notes as sentences
    excluded_columns: List[str] = []         # columns this attempt ran without
    cached_from: Optional[int] = None        # identical to this earlier attempt
    # a leak check: this record re-ran attempt `ablation_of` without `ablated_columns`
    ablation_of: Optional[int] = None
    ablated_columns: List[str] = []
    verdict: str = ""        # confirmed | cleared | declared_available | kept_by_policy | unresolved
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
    eligibility: str = "clean"      # clean | unverified | blocked
    eligibility_note: str = ""
