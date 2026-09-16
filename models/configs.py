from pydantic import BaseModel
from typing import List


class Baseline(BaseModel):
    strategy: str
    applicable: bool


class Configs(BaseModel):
    id: int = 0            # unused: the run id is assigned in code, not by the generator
    models: List[str]
    baseline: Baseline
    eval_matrics: List[str]
    early_stopping_patience: int
    max_tries: int
    improvement_delta: float
    improvement_mode: str
    improvement_metric: str
