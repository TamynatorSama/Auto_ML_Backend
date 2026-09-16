from pydantic import BaseModel
from typing import List, Optional


class SplitPlan(BaseModel):
    method: str
    test_size: float
    random_seed: int
    shuffle: bool
    time_column: Optional[str] = None
    group_column: Optional[str] = None
    stratify_column: Optional[str] = None
    drop_duplicates: bool
    cv_strategy: str
    cv_folds: int
    reason: str
    warnings: List[str] = []
